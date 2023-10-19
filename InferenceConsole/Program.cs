using ClaviRuntime;
using OpenCvSharp;
using System;
using System.Drawing;
using System.Text.Json;
using System.Security.Cryptography.X509Certificates;
using System.Diagnostics.Metrics;
using Newtonsoft.Json.Linq;

class Program
{
    static void Main(string[] args)
    {
        // Anomaly
        /*        string imagePath = "C:\\Users\\Beck\\Model\\model-test-lib\\anomaly\\dataset\\No-crop\\test03.png";
                string modelPath = "C:\\Users\\Beck\\Model\\model-test-lib\\anomaly\\model\\anoCLS.onnx";

                var anomaly = new Anomaly();
                anomaly.InitializeModel(modelPath);
                anomaly.Process(imagePath);
                List<AnomalyResults>? results = anomaly.resultsList;

                foreach (var r in results)
                {
                    Console.WriteLine("This image has a " +  r.Score + "% abnormality.");
                    Bitmap res = r.Heatmap;
                    res.Save("C:\\Users\\Beck\\Desktop\\original.jpg");
                }

                Console.WriteLine("Done");*/

        //Instance Segmentation
        /*        string imagePath = "C:\\Users\\Beck\\Pictures\\Original_Dataset\\Sample_instance\\IMG-0946.jpg";
                string modelPath = "C:\\Users\\Beck\\Model\\model-test-lib\\instance\\model\\Instance_Segment.onnx";

                var instanceSegmentation = new InstanceSegmentation();
                instanceSegmentation.InitializeModel(modelPath);
                instanceSegmentation.Process(imagePath);
                Bitmap resultImage = instanceSegmentation.imageResult;
                resultImage.Save("C:\\Users\\Beck\\Desktop\\IMG-0946_result.jpg");
                List<InstanceSegmentationResults> results = instanceSegmentation.resultsList;
                if (results != null)
                {
                    foreach (var r in results)
                    {
                        Console.WriteLine("Output for {0}", r.Id);
                        Console.WriteLine(r.Name);
                        Console.WriteLine(r.Score);
                    }
                }

                Console.WriteLine("Done");*/

        //Image Classification
        /*        string imagePath = "C:\\Users\\Beck\\Model\\model-test-lib\\classification\\tire\\image\\SRTTSW\\test01.png";
                string modelPath = "C:\\Users\\Beck\\Model\\model-test-lib\\classification\\tire\\model\\tire20230905.onnx";
                Mat inputImg = Cv2.ImRead(imagePath);

                var classify = new ImageClassification();
                classify.InitializeModel(modelPath);
                classify.Process(imagePath);
                List<ImageClassificationResults> results = classify.resultsList;
                if (results.Count != 0)
                {
                    foreach (var r in results)
                    {
                        Console.WriteLine(r.Name);
                        Console.WriteLine(r.Score);
                    }
                }*/

        //Object Detection
        //string imagePath = "C:\\Users\\Beck\\Pictures\\Original_Dataset\\Sample_Obj\\IMG-0970.jpg";
        //string modelPath = "C:\\Users\\Beck\\Downloads\\object detection\\end2end.onnx";
        string modelPath = "C:\\Users\\Beck\\Downloads\\NipponSteel\\ModelNipponSteel\\NipponSteel.onnx";

        var obj = new ObjectDetection();
        obj.InitializeModel(modelPath);
        //obj.Process(imagePath);
        //var resultImage = obj.imageResult;
        //List<ObjectDetectionResults>? results = obj.resultList;
/*        if (results != null)
        {
            foreach (var r in results)
            {
                Console.WriteLine("Output for index {0}", r.Index);
                Console.WriteLine(r.Label);
                Console.WriteLine(r.Score);
                Console.WriteLine(r.Box[0].ToString() + " | " + r.Box[1].ToString() + " | " + r.Box[2].ToString() + " | " + r.Box[3].ToString());
            }
        }*/

        string ImagePath = "C:\\Users\\Beck\\Downloads\\NipponSteel\\Data\\";
        foreach (string imageFileName in Directory.GetFiles(ImagePath, "*.jpg"))
        {
            obj.Process(imageFileName);
            var resultImage = obj.imageResult;
            string fileName = Path.GetFileNameWithoutExtension(imageFileName);
            resultImage.Save("C:\\Users\\Beck\\Downloads\\NipponSteel\\Result\\" + fileName + ".jpg");
        }

        //resultImage.Save("C:\\Users\\Beck\\Desktop\\results.jpg");
        //Console.WriteLine("Done");

        //Semantic Segmentation
        /*        string imagePath = "C:\\Users\\Beck\\Pictures\\Dataset\\RankI\\Rank1_01.JPG";
                string modelPath = "C:\\Users\\Beck\\Downloads\\semantic\\end2end.onnx";

                var semanticSegmentation = new SemanticSegmentation();
                semanticSegmentation.InitializeModel(modelPath);
                Bitmap resultImage = semanticSegmentation.Process(imagePath);
                List<SemanticSegmentationResults>? results = semanticSegmentation.resultsList;
                foreach (var r in results)
                {
                    Bitmap res = r.Mask;
                }
                resultImage.Save("C:\\Users\\Beck\\Desktop\\original.jpg");

                Console.WriteLine("Done");*/


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