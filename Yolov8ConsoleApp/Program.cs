
using System;
using System.Drawing;
using System.Text.Json;
using System.Security.Cryptography.X509Certificates;
using System.Diagnostics.Metrics;
using Yolov8Net;
using SixLabors.ImageSharp.Formats;
using System.Drawing.Drawing2D;
//using Newtonsoft.Json.Linq;

class Program
{
    static void Main(string[] args)
    {
        // Create new Yolov8 predictor, specifying the model (in ONNX format)
        // If you are using a custom trained model, you can provide an array of labels. Otherwise, the standard Coco labels are used.
        using var yolo = YoloV8Predictor.Create("C:\\Users\\Beck\\source\\repos\\Yolov8.Net\\test\\Yolov8net.test\\Assets\\yolov8m.onnx");

        // Provide an input image.  Image will be resized to model input if needed.
        using var img = SixLabors.ImageSharp.Image.Load("C:\\Users\\Beck\\source\\repos\\Yolov8.Net\\test\\Yolov8net.test\\Assets\\input.jpg");
        var predictions = yolo.Predict(img);

        // Draw your boxes
        System.Drawing.Image image = System.Drawing.Image.FromFile("C:\\Users\\Beck\\source\\repos\\Yolov8.Net\\test\\Yolov8net.test\\Assets\\input.jpg");
        using (Graphics graphics = Graphics.FromImage(image))
        {
            foreach (var pred in predictions)
            {
                var originalImageHeight = image.Height;
                var originalImageWidth = image.Width;

                var x = Math.Max(pred.Rectangle.X, 0);
                var y = Math.Max(pred.Rectangle.Y, 0);
                var width = Math.Min(originalImageWidth - x, pred.Rectangle.Width);
                var height = Math.Min(originalImageHeight - y, pred.Rectangle.Height);

                ////////////////////////////////////////////////////////////////////////////////////////////
                // *** Note that the output is already scaled to the original image height and width. ***
                ////////////////////////////////////////////////////////////////////////////////////////////

                // Bounding Box Text
                string text = $"{pred.Label.Name} [{pred.Score}]";
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;

                // Define Text Options
                Font drawFont = new Font("consolas", 11, FontStyle.Regular);
                System.Drawing.SizeF size = graphics.MeasureString(text, drawFont);
                SolidBrush fontBrush = new SolidBrush(System.Drawing.Color.Black);
                System.Drawing.Point atPoint = new System.Drawing.Point((int)x, (int)y - (int)size.Height - 1);

                // Define BoundingBox options
                Pen pen = new Pen(System.Drawing.Color.Yellow, 2.0f);
                SolidBrush colorBrush = new SolidBrush(System.Drawing.Color.Yellow);

                // Draw text on image 
                graphics.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width, (int)size.Height);
                graphics.DrawString(text, drawFont, fontBrush, atPoint);

                // Draw bounding box on image
                graphics.DrawRectangle(pen, x, y, width, height);
            }
            image.Save("C:\\Users\\Beck\\Desktop\\clip_temp_result\\output.jpg");

        }
    }
}