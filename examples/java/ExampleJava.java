import com.jcudbscan.JCUDBSCANJava;
import java.util.Random;

/**
 * Example demonstrating the use of the JCUDBSCANJava class.
 */
public class ExampleJava {
    
    public static void main(String[] args) {
        // Create an instance of the JCUDBSCANJava class
        JCUDBSCANJava cudbscan = new JCUDBSCANJava();
        
        // Check if CUDA is available
        boolean cudaAvailable = cudbscan.checkCudaAvailable();
        System.out.println("CUDA available: " + cudaAvailable);
        
        if (!cudaAvailable) {
            System.out.println("CUDA is not available. Please check your installation.");
            return;
        }
        
        // Create a random array of floats
        int size = 1_000_000;
        System.out.println("Creating random array with " + size + " elements...");
        
        float[] data = new float[size];
        Random random = new Random();
        for (int i = 0; i < size; i++) {
            data[i] = random.nextFloat();
        }
        
        // Run the CUDA function
        System.out.println("Running CUDA dummy function...");
        long startTime = System.currentTimeMillis();
        float[] result = cudbscan.cudaDummyFunction(data);
        long endTime = System.currentTimeMillis();
        
        // Verify the result
        boolean isCorrect = true;
        for (int i = 0; i < size; i++) {
            if (Math.abs(result[i] - data[i] * 2.0f) > 1e-5) {
                isCorrect = false;
                break;
            }
        }
        
        System.out.println("Execution time: " + (endTime - startTime) + " ms");
        System.out.println("Result is correct: " + isCorrect);
        
        // Show some results
        System.out.println("\nFirst 5 elements:");
        for (int i = 0; i < 5; i++) {
            System.out.printf("Input: %.6f, Output: %.6f, Expected: %.6f\n", 
                    data[i], result[i], data[i] * 2.0f);
        }
    }
}