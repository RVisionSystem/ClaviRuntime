using ClaviRuntime;
using OpenCvSharp;
using System;
using System.Drawing;
using System.Text.Json;
using System.Security.Cryptography.X509Certificates;
using System.Diagnostics.Metrics;
using Newtonsoft.Json.Linq;
using ClaviRuntime;

class Program
{
    static void Main(string[] args)
    {
        //Face Recognition 
        //string targetImagePath = "C:\\Users\\Beck\\Downloads\\aitest.jpg";
/*        string modelPath = "C:\\Users\\Beck\\Model\\model-test-lib\\face\\model\\face-recog22_9.onnx";
        var face = new FaceRecognition();
        face.InitializeModel(modelPath);
        //Bitmap resultImage = instanceSegmentation.process(imagePath);
        //face.Process(targetImagePath, targetImagePath);
        //List<FaceRecognitionResults>? results = face.resultsList;
        //face.CreateDB();

        // For testing
        string peoplePath = "C:\\Users\\Beck\\Model\\model-test-lib\\face\\";
        foreach (string peopleFileName in Directory.GetFiles(peoplePath, "*.txt"))
        {
            Console.WriteLine(peopleFileName);
            // file name without extension
            string fileName = Path.GetFileNameWithoutExtension(peopleFileName);
            string ImagePath = "C:\\Users\\Beck\\Model\\model-test-lib\\face\\" + fileName + "\\";
            foreach (string imageFileName in Directory.GetFiles(ImagePath, "*.jpg"))
            {
                face.Process(imageFileName, imageFileName);
            }

        }*/
        //resultImage.Save("C:\\Users\\Beck\\Desktop\\IMG-0946_result.jpg");
        //Console.WriteLine("Done");
        //face.ReadJSON();

    }
}

