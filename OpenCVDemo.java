import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class OpenCVDemo {

    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME);}

    public static void main(String[] args) {

        // Load an image
        Mat image = Imgcodecs.imread("img/img1.jpg", Imgcodecs.IMREAD_GRAYSCALE);
        Imgproc.resize(image, image, new Size(480, 640));

        // Convert the image to grayscale and blur it slightly
        //Imgproc.cvtColor(image, image, Imgproc.COLOR_GRAY2BGR);
        Imgproc.GaussianBlur(image, image, new Size(7, 7), 3,3);

        Imgproc.adaptiveThreshold(image, image, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 11, 2);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierathy = new Mat();
        Imgproc.findContours(image, contours, hierathy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        

        HighGui.namedWindow("test", HighGui.WINDOW_NORMAL);
        HighGui.imshow("test", image);

        HighGui.waitKey();
        System.exit(0);

    }

    // --- LINES DETECTION --- //
//    public static void main(String[] args) {
//
//        // Declare the output variables
//        Mat dst = new Mat(), cdst = new Mat();
//
//        // Load an image
//        Mat src = Imgcodecs.imread("img/img1.jpg", Imgcodecs.IMREAD_GRAYSCALE);
//        Imgproc.resize(src, src, new Size(480, 640));
//
//        // Edge detection
//        Imgproc.Canny(src, dst, 150, 500, 3, false);
//
//        // Copy edges to the images that will display the results in BGR
//        Imgproc.cvtColor(dst, cdst, Imgproc.COLOR_GRAY2BGR);
//
//        // Standard Hough Line Transform
//        Mat lines = new Mat(); // will hold the results of the detection
//        Imgproc.HoughLines(dst, lines, 1, Math.PI/180, 150); // runs the actual detection
//
//        // Draw the lines
//        for (int x = 0; x < lines.rows(); x++) {
//            double rho = lines.get(x, 0)[0],
//                    theta = lines.get(x, 0)[1];
//            double a = Math.cos(theta),
//                    b = Math.sin(theta);
//            double x0 = a*rho,
//                    y0 = b*rho;
//            Point pt1 = new Point(Math.round(x0 + 1000*(-b)), Math.round(y0 + 1000*(a)));
//            Point pt2 = new Point(Math.round(x0 - 1000*(-b)), Math.round(y0 - 1000*(a)));
//            Imgproc.line(cdst, pt1, pt2, new Scalar(0, 255, 0), 1, Imgproc.LINE_AA, 0);
//        }
//
//        // Show results
//        HighGui.namedWindow( "Source", HighGui.WINDOW_NORMAL );
//        HighGui.imshow("Source", src);
//        HighGui.namedWindow( "Detected Lines", HighGui.WINDOW_NORMAL );
//        HighGui.imshow("Detected Lines", cdst);
//
//        // Wait and Exit
//        HighGui.waitKey();
//        System.exit(0);
//
//    }


    private static void Harris(Mat Scene, Mat Object, int thresh) {

        // This function implements the Harris Corner detection. The corners at intensity > thresh
        // are drawn.
        Mat Harris_scene = new Mat();
        Mat Harris_object = new Mat();

        Mat harris_scene_norm = new Mat(), harris_object_norm = new Mat(), harris_scene_scaled = new Mat(), harris_object_scaled = new Mat();
        int blockSize = 9;
        int apertureSize = 5;
        double k = 0.1;
        Imgproc.cornerHarris(Scene, Harris_scene,blockSize, apertureSize,k);
        Imgproc.cornerHarris(Object, Harris_object, blockSize,apertureSize,k);

        Core.normalize(Harris_scene, harris_scene_norm, 0, 255, Core.NORM_MINMAX, CvType.CV_32FC1, new Mat());
        Core.normalize(Harris_object, harris_object_norm, 0, 255, Core.NORM_MINMAX, CvType.CV_32FC1, new Mat());

        Core.convertScaleAbs(harris_scene_norm, harris_scene_scaled);
        Core.convertScaleAbs(harris_object_norm, harris_object_scaled);

        for( int j = 0; j < harris_scene_norm.rows() ; j++){
            for( int i = 0; i < harris_scene_norm.cols(); i++){
                if ((int) harris_scene_norm.get(j,i)[0] > 10){
                    Imgproc.circle(harris_scene_scaled, new Point(i,j), 5 , new Scalar(0), 2 ,8 , 0);
                }
            }
        }

        for( int j = 0; j < harris_object_norm.rows() ; j++){
            for( int i = 0; i < harris_object_norm.cols(); i++){
                if ((int) harris_object_norm.get(j,i)[0] > 10){
                    Imgproc.circle(harris_object_scaled, new Point(i,j), 5 , new Scalar(0), 2 ,8 , 0);
                }
            }
        }
    }

}
