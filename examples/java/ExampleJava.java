import com.cudbscan.CUDBSCANJava;
import java.util.Random;

/**
 * Example demonstrating the use of the CUDBSCANJava class.
 */
public class ExampleJava {
    
    public static void main(String[] args) {
        // Create an instance of the CUDBSCANJava class
        CUDBSCANJava cudbscan = new CUDBSCANJava();
        
        // Check if CUDA is available
        boolean cudaAvailable = cudbscan.checkCudaAvailable();
        System.out.println("CUDA available: " + cudaAvailable);
        
        if (!cudaAvailable) {
            System.out.println("CUDA is not available.