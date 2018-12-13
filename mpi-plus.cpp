#include <string>
#include <iostream>
#include <mpi.h>




// ============================================================================
namespace mpi {
    class Communicator;
    class Request;
    class Status;

    static inline Communicator comm_world();
    constexpr int any_tag = MPI_ANY_TAG;
    constexpr int any_source = MPI_ANY_SOURCE;
}




// ============================================================================
class mpi::Request
{
public:
    Request() {}

    Request(const Request& other) = delete;

    Request(Request&& other)
    {
        buffer = std::move(other.buffer);
        request = other.request;
        other.request = MPI_REQUEST_NULL;
    }

    ~Request()
    {
        if (! is_null())
        {
            MPI_Cancel(&request);
            MPI_Request_free(&request);
        }
    }

    Request& operator=(const Request& other) = delete;

    Request& operator=(Request&& other)
    {
        buffer = std::move(other.buffer);
        request = other.request;
        other.request = MPI_REQUEST_NULL;
        return *this;        
    }

    bool is_null() const
    {
        return request == MPI_REQUEST_NULL;
    }

    bool test()
    {
        throw_if_null();
        int flag;
        MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
        return flag;
    }

    void wait()
    {
        throw_if_null();
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }

    const std::string& get()
    {
        wait();
        return buffer;
    }

    template <typename T>
    T get()
    {
        static_assert(std::is_standard_layout<T>::value, "type has non-standard layout");

        if (buffer.size() != sizeof(T))
        {
            throw std::logic_error("received message has wrong size for data type");   
        }

        wait();

        auto value = T();
        std::memcpy(&value, &buffer[0], sizeof(T));
        return value;
    }

private:
    // ========================================================================
    void throw_if_null()
    {
        if (is_null())
        {
            throw std::logic_error("method call on null request");
        }
    }
    friend class Communicator;
    MPI_Request request = MPI_REQUEST_NULL;
    std::string buffer;
};




// ============================================================================
/**
    This is a simple wrapper class around the MPI_Status struct. It 
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
     *              comm.isend("Message!", 0).wait();
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
        static_assert(std::is_standard_layout<T>::value, "type has non-standard layout");
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
        static_assert(std::is_standard_layout<T>::value, "type has non-standard layout");
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
        static_assert(std::is_standard_layout<T>::value, "type has non-standard layout");
        auto buf = recv(rank, tag);

        if (buf.size() != sizeof(T))
        {
            throw std::logic_error("received message has wrong size for data type");
        }

        auto value = T();
        std::memcpy(&value, &buf[0], sizeof(T));
        return value;
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

        if (comm.rank() == 0)
        {
            comm.send("Here is a message!", 1, 123);
            comm.send(3.14, 1, 124);
            comm.send("the", 1, 125);
            comm.send(20, 1, 126);
        }
        if (comm.rank() == 1)
        {
            std::cout << comm.recv(mpi::any_source, 123) << std::endl;
            std::cout << comm.recv<double>(mpi::any_source, 124) << std::endl;
            std::cout << comm.irecv(mpi::any_source, 125).get() << std::endl;
            std::cout << comm.irecv(mpi::any_source, 126).get<int>() << std::endl;
        }
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }

    MPI_Finalize();
    return 0;
}
