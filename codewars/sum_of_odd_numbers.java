/*
https://www.codewars.com/kata/55fd2d567d94ac3bc9000064

Given the triangle of consecutive odd numbers:

             1
          3     5
       7     9    11
   13    15    17    19
21    23    25    27    29
...

Calculate the sum of the numbers in the nth row of this triangle (starting at index 1) e.g.: (Input --> Output)

1 -->  1
2 --> 3 + 5 = 8
*/

import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

class RowSumOddNumbers {

  private static void printTriangle(int maxLevel) {
    var counter = new AtomicInteger(0);
    var currentLevel = new AtomicInteger(1);

    int totalOddNums = IntStream.range(1, maxLevel + 1).sum();
    IntStream
      .iterate(1, i -> i + 2)
      .limit(totalOddNums)
      .forEach(oddNum -> {
        if (currentLevel.get() == counter.getAndIncrement()) {
          System.out.println();
          currentLevel.incrementAndGet();
          counter.set(1);
        }
        System.out.print(oddNum + " ");
      });
    System.out.println();
  }

  public static int rowSumOddNumbers(int n) {
    int totalOddNums = IntStream.range(1, n + 1).sum();
    int rangeStart = 1 + 2 * (totalOddNums - n);
    return IntStream.iterate(rangeStart, i -> i + 2).limit(n).parallel().sum();
  }

  public static int rowSumOddNumbersSimple(int n) {
    return (int) Math.pow(n, 3);
  }

  public static void main(String[] args) {
    int arg = 5;

    System.out.println("Triangle:");
    printTriangle(arg);

    System.out.println("\nSum of row " + arg + ":");
    System.out.println(rowSumOddNumbers(arg));
    System.out.println(rowSumOddNumbersSimple(arg));
  }
}
