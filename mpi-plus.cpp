#include <string>
#include <iostream>
#include <vector>
#include <mpi.h>




// ============================================================================
namespace mpi {
    class Communicator;
    class Request;
    class Status;

    inline Communicator comm_world();
    constexpr int any_tag = MPI_ANY_TAG;
    constexpr int any_source = MPI_ANY_SOURCE;

    namespace detail {
        // template <typename T> inline int make_datatype_for(const T&);
    }
}

// template <> int mpi::detail::make_datatype_for<char>  (const char&)   { return MPI_CHAR; }
// template <> int mpi::detail::make_datatype_for<int>   (const int&)    { return MPI_INTEGER; }
// template <> int mpi::detail::make_datatype_for<float> (const float&)  { return MPI_FLOAT; }
// template <> int mpi::detail::make_datatype_for<double>(const double&) { return MPI_DOUBLE; }




// ============================================================================
/**
 * A thin RAII wrapper around the MPI_Request struct. This is a movable, but
 * non-copyable object. Keep it around on the stack in order to check on the
 * status of a non-blocking communication and retrieve the message content.
 * Requests are cancelled and deallocated (if necessary) when they go out of
 * scope.
 */
class mpi::Request
{
public:


    /**
     * Default constructor, creates a null request.
     */
    Request() {}


    /**
     * Request is a unique object, no copy's are permitted.
     */
    Request(const Request& other) = delete;


    /**
     * Move constructor. Steals ownership of the other.
     */
    Request(Request&& other)
    {
        buffer = std::move(other.buffer);
        request = other.request;
        other.request = MPI_REQUEST_NULL;
    }


    /**
     * Destructor. Cancels the request if one is pending. For this reason, the
     * request returned by non-blocking communications must be retained on the
     * stack somewhere in order for the operation not to be cancelled.
     */
    ~Request()
    {
        cancel();
    }


    /**
     * Copy assignment is not permitted.
     */
    Request& operator=(const Request& other) = delete;


    /**
     * Move assignment. Cancels the current request (if one is pending) and
     * steals ownership of the other.
     */
    Request& operator=(Request&& other)
    {
        cancel();
        buffer = std::move(other.buffer);
        request = other.request;
        other.request = MPI_REQUEST_NULL;
        return *this;        
    }


    /**
     * Cancel this request and reset its state to null.
     */
    void cancel()
    {
        if (! is_null())
        {
            MPI_Cancel(&request);
            MPI_Request_free(&request);
        }
    }


    /**
     * Return true if this request is null.
     */
    bool is_null() const
    {
        return request == MPI_REQUEST_NULL;
    }


    /**
     * Check to see whether the request has completed. If it has, this method
     * returns true and resets the request to a null state. If this method
     * returns true and the request was for a non-blocking receive operation,
     * the get() method can be called to retrieve the message content.
     */
    // bool test()
    // {
    //     int flag;
    //     MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
    //     return flag;
    // }


    /**
     * Check to see whether the request has completed. Like above, except
     * this const method will not free the request if it has completed.
     */
    bool is_ready() const
    {
        int flag;
        MPI_Request_get_status(request, &flag, MPI_STATUS_IGNORE);
        return flag;
    }


    /**
     * Block until the request is fulfilled. After this method returns, the
     * get() method can be called to retrieve the message content.
     */
    // void wait()
    // {
    //     MPI_Wait(&request, MPI_STATUS_IGNORE);
    // }


    /**
     * Block until the message is completed. Then deallocate the request, and
     * return the message content.
     */
    const std::string& get()
    {
        // wait();
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        return buffer;
    }


    /**
     * Return the message content formatted as the given data type.
     */
    template <typename T>
    T get()
    {
        static_assert(std::is_trivially_copyable<T>::value, "type is not trivially copyable");

        if (buffer.size() != sizeof(T))
        {
            throw std::logic_error("received message has wrong size for data type");   
        }

        // wait();

        MPI_Wait(&request, MPI_STATUS_IGNORE);

        auto value = T();
        std::memcpy(&value, &buffer[0], sizeof(T));
        return value;
    }


private:
    // ========================================================================
    friend class Communicator;
    MPI_Request request = MPI_REQUEST_NULL;
    std::string buffer;
};




// ============================================================================
/**
    This is a simple wrapper class around the MPI_Status struct.
*/
class mpi::Status
{
public:


    /**
     * Default-constructed status has is_null() == true.
     */
    Status() {}


    /**
     * Check for null-status. Null status is returned e.g. when iprobe does
     * not find any incoming messages:
     *
     *              if (comm.iprobe().is_null()) { }
     * 
     */
    bool is_null() const
    {
        return null;
    }


    /**
     * Return the number of bytes in the message described by this status.
     */
    int count()
    {
        if (is_null())
        {
            return 0;
        }

        int count;
        MPI_Get_count(&status, MPI_CHAR, &count);
        return count;
    }


    /**
     * Get the rank of the message's source.
     */
    int source() const
    {
        if (is_null())
        {
            return -1;
        }
        return status.MPI_SOURCE;
    }


    /**
     * Get the tag of the message.
     */
    int tag() const
    {
        if (is_null())
        {
            return -1;
        }
        return status.MPI_TAG;
    }


private:
    // ========================================================================
    friend class Communicator;
    Status(MPI_Status status) : status(status), null(false) {}
    MPI_Status status;
    bool null = true;
};




// ============================================================================
class mpi::Communicator
{
public:


    /**
     * Default constructor, gives you MPI_COMM_NULL.
     */
    Communicator()
    {
    }


    /**
     * Copy constructor, duplicates the communicator and respects RAII.
     */
    Communicator(const Communicator& other)
    {
        if (! other.is_null())
        {
            MPI_Comm_dup(other.comm, &comm);
        }
    }


    /**
     * Move constructor, sets the other comm back to null.
     */
    Communicator(Communicator&& other)
    {
        comm = other.comm;
        other.comm = MPI_COMM_NULL;
    }


    /**
     * Destructor, closes the communicator unless it was null.
     */
    ~Communicator()
    {
        close();
    }


    /**
     * Assignment operator: closes this communicator and duplicates the other
     * one, unless the other one is null in which case sets this one to null,
     * e.g. you can reset a communicator by writing
     *
     *              comm = Communicator();
     *
     */
    Communicator& operator=(const Communicator& other)
    {
        close();

        if (! other.is_null())
        {
            MPI_Comm_dup(other.comm, &comm);
        }
        return *this;
    }


    /**
     * Move assignment. Steals the other communicator.
     */
    Communicator& operator=(Communicator&& other)
    {
        close();

        comm = other.comm;
        other.comm = MPI_COMM_NULL;
        return *this;
    }


    /**
     * Close the communicator if it wasn't null.
     */
    void close()
    {
        if (! is_null())
        {
            MPI_Comm_free(&comm);
            comm = MPI_COMM_NULL;
        }
    }


    /**
     * Return true if the communicator is null.
     */
    bool is_null() const
    {
        return comm == MPI_COMM_NULL;
    }


    /**
     * Return the number of ranks in the communicator. This returns zero for a
     * null communicator (whereas I think MPI implementations typically
     * produce an error).
     */
    int size() const
    {
        if (is_null())
        {
            return 0;
        }

        int res;
        MPI_Comm_size(comm, &res);
        return res;
    }


    /**
     * Return the rank of the communicator. This returns -1 for a null
     * communicator (whereas I think MPI implementations typically produce an
     * error).
     */
    int rank() const
    {
        if (is_null())
        {
            return -1;
        }

        int res;
        MPI_Comm_rank(comm, &res);
        return res;
    }


    /**
     * Block all ranks in the communicator at this points.
     */
    void barrier() const
    {
        MPI_Barrier(comm);
    }


    /**
     * Probe for an incoming message and return its status. This method blocks
     * until there is an incoming message to probe.
     */
    Status probe(int rank=any_source, int tag=any_tag) const
    {
        MPI_Status status;
        MPI_Probe(rank, tag, comm, &status);
        return status;
    }


    /**
     * Probe for an incoming message and return its status. This method will
     * not block, but returns a null status if there was no message to probe.
     */
    Status iprobe(int rank=any_source, int tag=any_tag) const
    {
        MPI_Status status;
        int flag;
        MPI_Iprobe(rank, tag, comm, &flag, &status);

        if (! flag)
        {
            return Status();
        }
        return status;
    }


    /**
     * Blocking-receive a message with the given source and tag. Return the
     * data as a string.
     */
    std::string recv(int source=any_source, int tag=any_tag) const
    {
        auto status = probe(source, tag);
        auto buf = std::string(status.count(), 0);

        MPI_Recv(&buf[0], buf.size(), MPI_CHAR, source, tag, comm, MPI_STATUS_IGNORE);
        return buf;
    }


    /**
     * Non-blocking receive a message with the given source and tag. Return a
     * request object that can be queried for the completion of the receive
     * operation. Note that the request is cancelled if allowed to go out of
     * scope. You should keep the request somewhere, and call test(), wait(),
     * or get() in a little while.
     */
    Request irecv(int source=any_source, int tag=any_tag) const
    {
        auto status = iprobe(source, tag);

        if (status.is_null())
        {
            return Request();
        }
        auto buf = std::string(status.count(), 0);

        MPI_Request request;
        MPI_Irecv(&buf[0], buf.size(), MPI_CHAR, source, tag, comm, &request);

        Request res;
        res.buffer = std::move(buf);
        res.request = request;
        return res;
    }


    /**
     * Blocking-send a string to the given rank.
     */
    void send(std::string buf, int rank, int tag=0) const
    {
        MPI_Send(&buf[0], buf.size(), MPI_CHAR, rank, tag, comm);
    }


    /**
     * Non-blocking send a string to the given rank. Returns a request object
     * that can be tested for completion or waited on. Note that the request
     * is cancelled if allowed to go out of scope. Also keep in mind your MPI
     * implementation may have chosen to buffer your message internally, in
     * which case the request will have completed immediately, and the
     * cancellation will have no effect. Therefore it is advisable to keep the
     * returned request object, or at least do something equivalent to:
     *
     *              auto result = comm.isend("Message!", 0).get();
     *
     * Of course this would literally be a blocking send, but you get the
     * idea. In practice you'll probably store the request somewhere and check
     * on it after a while.
     */
    Request isend(std::string buf, int rank, int tag=0) const
    {
        Request res;
        res.buffer = buf;
        MPI_Isend(&res.buffer[0], buf.size(), MPI_CHAR, rank, tag, comm, &res.request);
        return res;
    }


    /**
     * Template version of a blocking send. You can pass any standard-layout
     * data type here.
     */
    template <typename T>
    void send(const T& value, int rank, int tag=0) const
    {
        static_assert(std::is_trivially_copyable<T>::value, "type is not trivially copyable");
        auto buf = std::string(sizeof(T), 0);
        std::memcpy(&buf[0], &value, sizeof(T));
        send(buf, rank, tag);
    }


    /**
     * Template version of a non-blocking send. You can pass any standard-layout
     * data type here.
     */
    template <typename T>
    Request isend(const T& value, int rank, int tag=0) const
    {
        static_assert(std::is_trivially_copyable<T>::value, "type is not trivially copyable");
        auto buf = std::string(sizeof(T), 0);
        std::memcpy(&buf[0], &value, sizeof(T));
        return isend(buf, rank, tag);
    }


    /**
     * Template version of a blocking receive. You can pass any
     * standard-layout data type here.
     */
    template <typename T>
    T recv(int rank, int tag=0) const
    {
        static_assert(std::is_trivially_copyable<T>::value, "type is not trivially copyable");
        auto buf = recv(rank, tag);

        if (buf.size() != sizeof(T))
        {
            throw std::logic_error("received message has wrong size for data type");
        }

        auto value = T();
        std::memcpy(&value, &buf[0], sizeof(T));
        return value;
    }


    /**
     * Execute an all-to-all communication with character-based data. Each
     * rank sends the character at index i to rank i. The return value at
     * index j contains the character received from rank j. More generally if
     * the send buffer size divides the comm size N times, then N characters
     * are send to and received from each rank.
     */
    std::string all_to_all(const std::string& sendbuf) const
    {
        if (sendbuf.size() % size() != 0)
        {
            throw std::invalid_argument("all_to_all send buffer must be divisible by the comm size");
        }

        auto recvbuf = std::string(sendbuf.size(), 0);

        MPI_Alltoall(
            &sendbuf[0], sendbuf.size() / size(), MPI_CHAR,
            &recvbuf[0], recvbuf.size() / size(), MPI_CHAR, comm);

        return recvbuf;
    }


    /**
     * Same as above, except the data type is any standard layout data rather
     * than char.
     */
    template <typename T>
    std::vector<T> all_to_all(const std::vector<T>& sendbuf) const
    {
        static_assert(std::is_trivially_copyable<T>::value, "type is not trivially copyable");

        if (sendbuf.size() != size())
        {
            throw std::invalid_argument("all_to_all send buffer must equal the comm size");
        }

        auto recvbuf = std::vector<T>(sendbuf.size(), T());

        MPI_Alltoall(
            &sendbuf[0], sendbuf.size() / size() * sizeof(T), MPI_CHAR,
            &recvbuf[0], recvbuf.size() / size() * sizeof(T), MPI_CHAR, comm);

        return recvbuf;
    }


    /**
     * Execute an all-gather communication with data of the given scalar type.
     * The returned vector contains the value provided by process j at int
     * j-th index.
     */
    template <typename T>
    std::vector<T> all_gather(const T& value) const
    {
        static_assert(std::is_trivially_copyable<T>::value, "type is not trivially copyable");

        auto recvbuf = std::vector<T>(size(), T());
        MPI_Allgather(&value, sizeof(T), MPI_CHAR, &recvbuf[0], sizeof(T), MPI_CHAR, comm);
        return recvbuf;
    }


    /**
     * Execute an all-gather-v communication. This is a generalization of the
     * above, where each rank broadcasts to all others a container of items.
     * The size of the container to be broadcasted need not be the same on
     * every rank. The container broadcasted by rank j is returned in the j-th
     * index of the vector returned by this function.
     */
    template <typename T>
    std::vector<std::vector<T>> all_gather(const std::vector<T>& values) const
    {
        static_assert(std::is_trivially_copyable<T>::value, "type is not trivially copyable");

        auto recvcounts = all_gather<int>(values.size());
        auto displs = std::vector<int>();
        std::size_t last = 0;

        for (auto count : recvcounts)
        {
            displs.push_back(last);
            last += count;
        }
        auto recvbuf = std::vector<T>(last);

        MPI_Allgatherv(
            &values[0], values.size(), MPI_CHAR,
            &recvbuf[0], &recvcounts[0], &displs[0], MPI_CHAR, comm);

        auto res = std::vector<std::vector<T>>(size());
        auto recv = recvbuf.begin();

        for (int i = 0; i < res.size(); ++i)
        {
            res[i].resize(recvcounts[i]);

            for (int j = 0; j < res[i].size(); ++j)
            {
                res[i][j] = *recv++;
            }
        }
        return res;
    }


private:
    // ========================================================================
    friend Communicator comm_world();
    MPI_Comm comm = MPI_COMM_NULL;
};




// ============================================================================
mpi::Communicator mpi::comm_world()
{
    Communicator res;
    MPI_Comm_dup(MPI_COMM_WORLD, &res.comm);
    return res;
}




// ============================================================================
int main()
{
    MPI_Init(0, nullptr);

    try {
        auto comm = mpi::comm_world();

        // if (comm.rank() == 0)
        // {
        //     comm.send("Here is a message!", 1, 123);
        //     comm.send(3.14, 1, 124);
        //     comm.send("the", 1, 125);
        //     comm.send(20, 1, 126);
        // }
        // if (comm.rank() == 1)
        // {
        //     std::cout << comm.recv(mpi::any_source, 123) << std::endl;
        //     std::cout << comm.recv<double>(mpi::any_source, 124) << std::endl;
        //     std::cout << comm.irecv(mpi::any_source, 125).get() << std::endl;
        //     std::cout << comm.irecv(mpi::any_source, 126).get<int>() << std::endl;
        // }


        // if (comm.rank() == 0)
        // {
        //     std::cout << "Rank 0 all-to-all: " << comm.all_to_all("00") << std::endl;
        // }
        // if (comm.rank() == 1)
        // {
        //     std::cout << "Rank 1 all-to-all: " << comm.all_to_all("11") << std::endl;
        // }


        auto res = comm.all_gather(comm.rank());
        auto ses = comm.all_gather(std::vector<int>(comm.rank()));

        for (int i = 0; i < comm.size(); ++i)
        {
            if (i == comm.rank())
            {
                std::cout << "rank " << i << ": " << res[i] << " " << ses[i].size() << std::endl;                
            }
            comm.barrier();
        }

    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    MPI_Finalize();
    return 0;
}
