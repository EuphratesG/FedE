import threading

def get_thread_count():
    return threading.active_count()

if __name__ == "__main__":
    thread_count = get_thread_count()
    print(f"Active thread count: {threading.active_count()}")