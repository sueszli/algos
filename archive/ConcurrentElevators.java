// minimal thread-pool-executor service / scheduler
// 
// Problem:
// 
// - There are 7 elevators in a building with 55 floors.
// - Each floor has exactly one button to request an elevator.
// - Elevators have the capacity of carrying an arbitrary, given number of people.
// - Elevators don't change direction and don't stop mid-trip.
// - People can only choose their destination freely, when they are on the ground-floor:
//     - up: ground-floor [0] − can only travel to non-ground-floors [1;54]
//     - down: non-ground-floor [1;54] − can only travel to the ground-floor [0]
// - The requests are handled globally.
// - The elevator-scheduler assigns a chosen request to an elevator based on some arbitrary algorithm.
// - If there are no elevators available, the request is queued up.
// 
// Based on: https://gitlab.com/niklaswimmer/dc-tower-elevator-challange/-/blob/main/app/src/main/java/me/nikx/dctower/TowerController.java
//
// $ java --version # must be 21 or higher
// $ javac ConcurrentElevators.java
// $ java ConcurrentElevators
// $ clang-format -i *.java
//

import java.util.Comparator;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;
import java.util.stream.*;

static final int NUM_FLOORS = 55;
static final int NUM_ELEVATORS = 7;
static final int QUEUE_SIZE = 10;
static final int DELAY = 1000;

record Request(int src, int dst) {
  static final Request POISON = new Request(-1, -1); // shutdown signal
  static final Predicate<Request> isValid = r
      -> r.src >= 0 && r.dst >= 0 && r.src != r.dst && r.src <= NUM_FLOORS &&
             r.dst <= NUM_FLOORS && (r.src == 0 || r.dst == 0);
}

record Scheduler(BlockingQueue<Request> queue,
                 ConcurrentHashMap<Integer, Elevator> elevators)
    implements Runnable {
  Scheduler() {
    this(new ArrayBlockingQueue<>(QUEUE_SIZE), new ConcurrentHashMap<>());
  }
  boolean receiveRequest(Request r) {
    return Request.isValid.test(r) && queue.size() < QUEUE_SIZE &&
        queue.offer(r) &&
        log("scheduler: received [" + r.src + " -> " + r.dst + "]");
  }
  void start() { new Thread(this, "scheduler").start(); }
  void shutdown() throws InterruptedException { queue.put(Request.POISON); }

  public void run() {
    log("scheduler: initializing elevators...");
    IntStream.range(0, NUM_ELEVATORS)
        .mapToObj(_ -> new Elevator())
        .peek(e -> new Thread(e, String.format("%02d", e.id)).start())
        .forEach(e -> elevators.put(e.id, e));

    log("scheduler: starting scheduler loop...");
    Stream
        .generate(() -> {
          try {
            return queue.take();
          } catch (InterruptedException e) {
            throw new RuntimeException(e);
          }
        })
        .takeWhile(r -> !r.equals(Request.POISON))
        .forEach(r -> {
          try {
            elevators.values()
                .stream()
                .min(Comparator.comparingInt(Elevator::numAssignedRequests))
                .orElseThrow()
                .assign(r);
          } catch (InterruptedException e) {
            throw new RuntimeException(e);
          }
        });

    log("scheduler: shutting down...");
    elevators.values().forEach(Elevator::shutdown);
    elevators.clear();
    queue.clear();

    log("scheduler: shut down successfully");
  }
}

record Elevator(int id, BlockingQueue<Request> queue) implements Runnable {
  static final AtomicInteger ID = new AtomicInteger(0);
  Elevator() {
    this(ID.getAndIncrement(), new ArrayBlockingQueue<>(QUEUE_SIZE));
  }
  int numAssignedRequests() { return queue.size(); }
  void assign(Request r) throws InterruptedException { queue.put(r); }
  void shutdown() {
    try {
      queue.put(Request.POISON);
    } catch (InterruptedException e) {
      throw new RuntimeException(e);
    }
  }

  public void run() {
    Stream
        .generate(() -> {
          try {
            return queue.take();
          } catch (InterruptedException e) {
            throw new RuntimeException(e);
          }
        })
        .takeWhile(r -> !r.equals(Request.POISON))
        .forEach(r -> {
          try {
            Thread.sleep(DELAY);
            System.out.println("elevator " + id + ": resolved [" + r.src +
                               " -> " + r.dst + "]");
          } catch (InterruptedException e) {
            throw new RuntimeException(e);
          }
        });
    System.out.println("elevator " + id + ": shut down successfully");
    queue.clear();
  }
}

static boolean log(String msg) {
  System.out.println(msg);
  return true;
}

void main(String[] args) throws InterruptedException {
  var scheduler = new Scheduler();
  Stream
      .generate(()
                    -> new Request((int)(Math.random() * NUM_FLOORS),
                                   (int)(Math.random() * NUM_FLOORS)))
      .limit(QUEUE_SIZE)
      .filter(scheduler::receiveRequest)
      .count();
  scheduler.start();
  Thread.sleep(1000);
  scheduler.shutdown();
}
