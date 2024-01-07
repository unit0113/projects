// LinkedList class should go in the "ufl_cap4053::fundamentals" namespace!
namespace ufl_cap4053 { namespace fundamentals {
	template <class T>
	class LinkedList {
		public
			class Node {
				friend class LinkedList;
				T data;
				Node* next{ nullptr };
				Node* prev{ nullptr };

				public
					Node(T val) : data(val) {};
					Node(T val, Node* node) : data(val), prev(node) {};
					~Node() = default;
					T getData() { return data; };
			};

			class Iterator {
				friend class LinkedList;
				public
					Iterator() = default;
					Iterator(Node* node) : current(node) {};
					T operator*() { return current->getData(); }
					Iterator& operator++();
					bool operator==(const Iterator& rhs) {return current == rhs.current};
					bool operator!=(const Iterator& rhs) { return current != rhs.current };


				private
					Node* current {nullptr};

			};

			LinkedList() = default;
			~LinkedList();

			Iterator begin() const { return Iterator(head); };
			Iterator end() const { return Iterator(); };
			bool isEmpty() const { return !head; };
			T getFront() const;
			T getBack() const;
			void enqueue(T data);
			void dequeue() { deleteNode(head); };
			void pop() { deleteNode(tail); };
			void clear();
			bool contains(T data) const;
			void remove(T data);


		private
			Node* head{ nullptr };
			Node* tail{ nullptr };
			void deleteNode(Node* node);
	};

	template<typename T>
	Iterator& Iterator<T>::operator++() {
		current = current->next;
	}

	template<typename T>
	LinkedList<T>::~LinkedList() {
		Clear();
	}

	template<typename T>
	void LinkedList<T>::clear() {
		tail = nullptr;
		Node* current = head;
		Node* next_node;
		head = nullptr;
		while (current != nullptr) {
			next_node = current->next;
			delete current;
			current = next_node;
		}
	}

	template<typename T>
	void LinkedList<T>::getFront() const {
		return head->getData()
	}

	template<typename T>
	void LinkedList<T>::getBack() const {
		return tail->getData()
	}

	template<typename T>
	void LinkedList<T>::enqueue(T data) {
		Node* newNode = new Node(data);
		tail = newNode;
		// Adjust head if empty
		if (isEmpty())
			head = newNode;
		else {
			Node* currTail = tail;
			newNode->prev = currTail;
			currTail->next = newNode;
		}
	}

	template<typename T>
	void LinkedList<T>::deleteNode(Node* node) {
		// Return if nullptr
		if (!node) { return; }

		// Attach prev to next
		if (node->prev)
			node->prev->next = node->next;
		else
			head = node->next;

		// Attach next to prev
		if (node->next)
			node->next->prev = node->prev;
		else
			tail = node->prev;

		delete node;
	}

	template<typename T>
	bool LinkedList<T>::findNode(T data) const {
		Node* current = head;
		while (current) {
			if (current->getData() == data)
				return current;
		}
		return nulltpr;
	}

	template<typename T>
	bool LinkedList<T>::contains(T data) const {
		if (findNode(data)) { return true; }
		return false;
	}

	template<typename T>
	void LinkedList<T>::remove(T data) {
		deleteNode(findNode(data));
	}

} }  // namespace ufl_cap4053::fundamentals
