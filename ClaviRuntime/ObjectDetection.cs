using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using Newtonsoft.Json;
using OpenCvSharp.Dnn;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Net.Mime.MediaTypeNames;
using SixLabors.ImageSharp;
using OpenCvSharp.WpfExtensions;
using System.Drawing.Imaging;

namespace ClaviRuntime
{
    public class ObjectDetection : IDisposable
    {
        private InferenceSession? SESS;
        public Bitmap? imageResult;
        public List<ObjectDetectionResults>? resultList;
        private Dictionary<string, string>? LABEL_LIST;
        private OpenCvSharp.Size PAD_SIZE;
        private OpenCvSharp.Size ACTUAL_SIZE;
        private OpenCvSharp.Size REQUIRED_SIZE;
        private OpenCvSharp.Size RESIZED_IMAGE;
        private int[]? DIMENSION;

        public void InitializeModel(string modelPath)
        {
            var option = new SessionOptions();
            option.GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL;
            option.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            SESS = new InferenceSession(modelPath, option);
        }

        public void Process(string imagePath, float threshold = 0.5F, float nmsThresh = 0.4f)
        {
            resultList = new List<ObjectDetectionResults>();
            Mat image = Cv2.ImRead(imagePath);
            //Check if the model is initialized
            try
            {
                // Setup inputs and outputs
                var inputMeta = SESS.InputMetadata;
                var inputName = inputMeta.Keys.ToArray()[0];
                DIMENSION = inputMeta[inputName].Dimensions;
                var customMeta = SESS.ModelMetadata.CustomMetadataMap;
                //New model customMeta.ToArray()[9].Value;
                LABEL_LIST = JsonConvert.DeserializeObject<Dictionary<string, string>>(customMeta.ToArray()[0].Value);

                // WIDTH = DIMENSION_SIZE[3]; HEIGHT = DIMENSION_SIZE[2];
                REQUIRED_SIZE = new OpenCvSharp.Size(DIMENSION[3], DIMENSION[2]);
                ACTUAL_SIZE = new OpenCvSharp.Size(image.Width, image.Height);
                RESIZED_IMAGE = GetResizedImage(ACTUAL_SIZE, REQUIRED_SIZE);

                var pallete = GenPalette(LABEL_LIST.Count);

                var processedImg = Preprocessing(image, REQUIRED_SIZE);
                var input = new DenseTensor<float>(MatToList(processedImg), new[] { DIMENSION[0], DIMENSION[1], DIMENSION[2], DIMENSION[3] });

                var inputs = new List<NamedOnnxValue>{ NamedOnnxValue.CreateFromTensor(inputName, input) };

                var results = SESS.Run(inputs);
                resultList = Postprocessing(results);

                if (resultList.Count != 0)
                {
                    //Draw bounding box on image without padding.
                    //Scalling back to original size (remove padding from rectangle size * scaling)
                    Bitmap a = BuildResultImage(image, resultList);
                    imageResult = a;

                }
            }
            catch (Exception e)
            {
                Console.WriteLine("Model not initialized");
                Console.WriteLine(e.Message);
                using (var ms = image.ToMemoryStream())
                {
                    imageResult = (Bitmap)System.Drawing.Image.FromStream(ms);
                }
            }
        }
        public static List<List<float>> GetCandidate(float[] pred, int[] pred_dim, float pred_thresh = 0.25f)
        {
            List<List<float>> candidate = new List<List<float>>();
            for (int batch = 0; batch < pred_dim[0]; batch++)
            {
                for (int cand = 0; cand < pred_dim[1]; cand++)
                {
                    int score = 4;//Default 4  // object ness score
                    int idx1 = (batch * pred_dim[1] * pred_dim[2]) + cand * pred_dim[2];
                    int idx2 = idx1 + score;
                    var value = pred[idx2];

                    List<float> tmp_value = new List<float>();
                    for (int i = 0; i < pred_dim[2]; i++)
                    {
                        int sub_idx = idx1 + i;
                        tmp_value.Add(pred[sub_idx]);
                    }
                    candidate.Add(tmp_value);
                }
            }
            return candidate;
        }
        static Vec3b[] GenPalette(int classes)
        {
            Random rnd = new Random(classes);
            Vec3b[] palette = new Vec3b[classes];
            for (int i = 0; i < classes; i++)
            {
                byte v1 = (byte)rnd.Next(0, 255);
                byte v2 = (byte)rnd.Next(0, 255);
                byte v3 = (byte)rnd.Next(0, 255);
                palette[i] = new Vec3b(v1, v2, v3);
            }
            return palette;
        }
        private static float[] MatToList(Mat mat)
        {
            var ih = mat.Height;
            var iw = mat.Width;
            var chn = mat.Channels();
            unsafe
            {
                return Create((float*)mat.DataPointer, ih, iw, chn);
            }
        }
        private unsafe static float[] Create(float* ptr, int ih, int iw, int chn)
        {
            float[] array = new float[chn * ih * iw];

            for (int y = 0; y < ih; y++)
            {
                for (int x = 0; x < iw; x++)
                {
                    for (int c = 0; c < chn; c++)
                    {
                        var idx = (y * chn) * iw + (x * chn) + c;
                        var idx2 = (c * iw) * ih + (y * iw) + x;
                        array[idx2] = ptr[idx];
                    }
                }
            }
            return array;
        }
        public static Mat AddPadding(Mat src, int top, int bottom, int left, int right, BorderTypes borderTypes, Scalar value)
        {
            Mat dst = new Mat();
            Cv2.CopyMakeBorder(src, dst, top, bottom, left, right, borderTypes, value);
            return dst;
        }

        //Remove padding
        private static Mat RemovePadding(Mat paddedImage, OpenCvSharp.Size padSize)
        {
            int top = padSize.Height/2;
            int bottom = padSize.Height/2;
            int left = padSize.Width/2;
            int right = padSize.Width / 2;
            Rect roi = new Rect(left, top, paddedImage.Width - left - right, paddedImage.Height - top - bottom);
            Cv2.Rectangle(paddedImage, new OpenCvSharp.Point(left, top), new OpenCvSharp.Point(paddedImage.Width - left - right, paddedImage.Height - top - bottom),
                                    new Scalar(0, 0, 0), -1);
            //Mat dst = new Mat(paddedImage, roi);
            return paddedImage;
        }

        //Calculate scaling
        private static OpenCvSharp.Size GetScaling(OpenCvSharp.Size originalSize, OpenCvSharp.Size targetSize)
        {
            float widthRatio = (float)targetSize.Width / (float)originalSize.Width;
            float heightRatio = (float)targetSize.Height / (float)originalSize.Height;
            float ratio = Math.Min(heightRatio, widthRatio);

            int newWidth = (int)(originalSize.Width * ratio);
            int newHeight = (int)(originalSize.Height * ratio);

            OpenCvSharp.Size sizeToResize = new OpenCvSharp.Size(newWidth, newHeight);

            return sizeToResize;
        }

        //Resize image
        private static OpenCvSharp.Size GetResizedImage(OpenCvSharp.Size actualSize, OpenCvSharp.Size requiredSize)
        {
            float widthRatio = (float)requiredSize.Width / (float)actualSize.Width;
            float heightRatio = (float)requiredSize.Height / (float)actualSize.Height;
            float ratio = Math.Min(heightRatio, widthRatio);

            // Calculate the new width and height based on the ratio
            int newWidth = (int)(actualSize.Width * ratio);
            int newHeight = (int)(actualSize.Height * ratio);

            OpenCvSharp.Size newSize = new OpenCvSharp.Size(newWidth, newHeight);

            return newSize;
        }

        private static Mat ResizeImage(Mat inputImage, OpenCvSharp.Size requiredSize)
        {
            float widthRatio = (float)requiredSize.Width / (float)inputImage.Width;
            float heightRatio = (float)requiredSize.Height / (float)inputImage.Height;
            float ratio = Math.Min(heightRatio, widthRatio);

            // Calculate the new width and height based on the ratio
            int newWidth = (int)(inputImage.Width * ratio);
            int newHeight = (int)(inputImage.Height * ratio);

            OpenCvSharp.Size newSize = new OpenCvSharp.Size(newWidth, newHeight);
            Mat resizedImage = inputImage.Resize(newSize);
            resizedImage.ConvertTo(resizedImage, MatType.CV_32FC3);

            return resizedImage;
        }

        //Get padding
        private static OpenCvSharp.Size GetPaddingSize(OpenCvSharp.Size requiredSize, OpenCvSharp.Size newSize)
        { 
            int TB = (requiredSize.Height - newSize.Height) / 2;
            int LR = (requiredSize.Width - newSize.Width) / 2;

            OpenCvSharp.Size paddingSize = new OpenCvSharp.Size(LR, TB);
            
            return paddingSize;
        }

        private static Mat Preprocessing(Mat inputMat, OpenCvSharp.Size requiredSize)
        {
            //Resize
            // Calculate the ratio of the new size to the original size
            float widthRatio = (float)requiredSize.Width / (float)inputMat.Width;
            float heightRatio = (float)requiredSize.Height / (float)inputMat.Height;
            float ratio = Math.Min(heightRatio, widthRatio);

            // Calculate the new width and height based on the ratio
            int newWidth = (int)(inputMat.Width * ratio);
            int newHeight = (int)(inputMat.Height * ratio);

            OpenCvSharp.Size newSize = new OpenCvSharp.Size(newWidth, newHeight);

            //Resize with aspect ratio
            Mat resizedImage = inputMat.Resize(newSize); 
            resizedImage.ConvertTo(resizedImage, MatType.CV_32FC3);
            //Calculate the top left bottom right padding
            OpenCvSharp.Size paddingSize = GetPaddingSize(requiredSize, newSize);
            int TB = paddingSize.Height;
            int LR = paddingSize.Width;

            //Padding
            Mat padded = AddPadding(resizedImage, TB, TB, LR, LR, BorderTypes.Constant, new Scalar(114, 114, 114));

            return padded;
        
        }

        private static List<ObjectDetectionResults> Postprocessing(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results)
        {
            List<ObjectDetectionResults> resList = new List<ObjectDetectionResults>();
            var resultsArray = results.ToArray();
            var pred_value = resultsArray[0].AsEnumerable<float>().ToArray();
            var pred_dim = resultsArray[0].AsTensor<float>().Dimensions.ToArray();
            var label_value = resultsArray[1].AsEnumerable<Int64>().ToArray();
            var candidate = GetCandidate(pred_value, pred_dim, 0.5F);
            if (candidate.Count != 0)
            {
                //NMS
                List<Rect> bboxes = new List<Rect>();
                List<float> confidences = new List<float>();
                for (int i = 0; i < candidate.Count; i++)
                {
                    Rect box = new Rect((int)candidate[i][0], (int)candidate[i][1],
                       (int)(candidate[i][2] - candidate[i][0]), (int)(candidate[i][3] - candidate[i][1]));
                    bboxes.Add(box);
                    confidences.Add(candidate[i][4]);
                }
                int[] indices;
                CvDnn.NMSBoxes(bboxes, confidences, 0.5F, 0.4F, out indices);

                if (indices != null)
                {
                    for (int ids = 0; ids < indices.Length; ids++)
                    {
                        int idx = indices[ids];
                        var confi = candidate[idx][4];

                        var Xmin = candidate[idx][0];
                        var Ymin = candidate[idx][1];
                        var Xmax = candidate[idx][2];
                        var Ymax = candidate[idx][3];

                        resList.Add(new ObjectDetectionResults(ids, idx.ToString(), confi, new float[] { Xmin, Ymin, Xmax, Ymax }));

                    }
                }
            }
            return resList;
        }

        //Convert Mat to Bitmap
        private static Bitmap MatToBitmap(Mat mat)
        {
            using (var ms = mat.ToMemoryStream())
            {
                return (Bitmap)System.Drawing.Image.FromStream(ms);
            }
        }

        private Bitmap BuildResultImage(Mat InputImage, List<ObjectDetectionResults> resultList)
        {
            var pallete = GenPalette(LABEL_LIST.Count);
            string[] brushArray = typeof(System.Drawing.Color).GetProperties().Select(c => c.Name).ToArray();

            System.Drawing.Color colorBox = System.Drawing.Color.FromName(brushArray[0]);
            //Convert Mat to Bitmap
            Bitmap imageToDraw = MatToBitmap(InputImage);
            //Deaw graphics
            using (var g = Graphics.FromImage(imageToDraw))
            {

                if (resultList != null)
                {
                    foreach (var res in resultList)
                    {

                        //div btw the resized image and the padded image
                        //RESIZED_IMAGE + PAD_SIZE;
                        var dw2 = ACTUAL_SIZE.Width / (float)RESIZED_IMAGE.Width;
                        var dh2 = ACTUAL_SIZE.Height / (float)RESIZED_IMAGE.Height;
                        PAD_SIZE = GetPaddingSize(REQUIRED_SIZE, RESIZED_IMAGE);

                        var l = ((res.Box[0] - PAD_SIZE.Width) * dw2);
                        var t = ((res.Box[1] - PAD_SIZE.Height) * dh2);
                        var r = ((res.Box[2] - PAD_SIZE.Width) * dw2);
                        var b = ((res.Box[3] - PAD_SIZE.Height) * dh2);

                        Pen blackPen = new Pen(System.Drawing.Color.Red, 10);
                        g.DrawRectangle(blackPen, l, t, r-l, b-t);

                    }
                }
            }

            return imageToDraw;
        }

        public void Dispose()
        {
            SESS?.Dispose();
            SESS = null;
        }



    }
}
