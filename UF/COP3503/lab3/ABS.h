#include <stdexcept>


template<typename T>
class ABS {
    public:
        ABS();
        ABS(int capacity);
        ABS(int capacity, float scale_factor);
        ABS(const ABS& other);
        ABS& operator=(const ABS& other);
        ~ABS();
        void push(T item);
        T peek() const;
        T pop();
        unsigned int getSize() const;
        unsigned int getMaxCapacity() const;
        T* getData();
        unsigned int getTotalResizes() const;

    private:
        unsigned int m_size;
        unsigned int m_capacity;
        T* m_data;
        unsigned int total_resizes {};
        static float c_scale_factor;
        void copy(const ABS& other);
        void increase_capacity();
        void decrease_capacity();

};

template<typename T>
float ABS<T>::c_scale_factor;

template<typename T>
ABS<T>::ABS() {
    m_size = 0;
    m_capacity = 1;
    m_data = new T[m_capacity];
    c_scale_factor = 2.0f;
}

template<typename T>
ABS<T>::ABS(int capacity) {
    m_size = 0;
    m_capacity = capacity;
    m_data = new T[m_capacity];
    c_scale_factor = 2.0f;
}

template<typename T>
ABS<T>::ABS(int capacity, float scale_factor) {
    m_size = 0;
    m_capacity = capacity;
    m_data = new T[m_capacity];
    c_scale_factor = scale_factor;
}

template<typename T>
ABS<T>::ABS(const ABS& other) {
    copy(other);
}

template<typename T>
ABS<T>& ABS<T>::operator=(const ABS& other) {
    copy(other);
    return *this;
}

template<typename T>
void ABS<T>::copy(const ABS& other) {
    delete[] m_data;
    m_size = other.m_size;
    m_capacity = other.m_capacity;
    m_data = new T[m_capacity];
    // Deep copy of array data
    for (unsigned int i{}; i < m_capacity; ++i) {
        m_data[i] = other.m_data[i];
    }
}

template<typename T>
ABS<T>::~ABS() {
    delete[] m_data;
}

template<typename T>
void ABS<T>::push(T item) {
    // Check if stack is full, and resize if yes
    if (m_size == m_capacity) {
        increase_capacity();
    }

    m_data[m_size] = item;
    ++m_size;
}

template<typename T>
void ABS<T>::increase_capacity() {
    T* new_data = new T[static_cast<int>(m_capacity * c_scale_factor)];
    // Deep copy of array data
    for (unsigned int i{}; i < m_capacity; ++i) {
        new_data[i] = m_data[i];
    }
    m_capacity *= c_scale_factor;
    delete[] m_data;
    m_data = new_data;
    ++total_resizes;
}

template<typename T>
T ABS<T>::peek() const {
    if (m_size == 0) {
        throw std::runtime_error("Stack is empty");
    }
    return m_data[m_size-1];
}

template<typename T>
T ABS<T>::pop() {
    if (m_size == 0) {
        throw std::runtime_error("Stack is empty");
    }
    T value = m_data[--m_size];

    // Check if capacity is too small, and resize if yes
    if ((static_cast<float>(m_capacity) -1) / m_size >= c_scale_factor) {
        decrease_capacity();
    }
    return value;
}

template<typename T>
void ABS<T>::decrease_capacity() {
    // If min size
    if (m_capacity == 1) {
        return;
    }

    m_capacity /= c_scale_factor;
    T* new_data = new T[m_capacity];
    // Deep copy of array data
    for (unsigned int i{}; i < m_capacity; ++i) {
        new_data[i] = m_data[i];
    }
    
    delete[] m_data;
    m_data = new_data;
    ++total_resizes;
}

template<typename T>
unsigned int ABS<T>::getSize() const {
    return m_size;
}

template<typename T>
unsigned int ABS<T>::getMaxCapacity() const {
    return m_capacity;
}

template<typename T>
T* ABS<T>::getData() {
    return m_data;
}

template<typename T>
unsigned int ABS<T>::getTotalResizes() const {
    return total_resizes;
}