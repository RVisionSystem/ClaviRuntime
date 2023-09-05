using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ClaviRuntime
{
    public class Anomaly : IDisposable
    {
        private InferenceSession? sess;
        public List<AnomalyResults>? resultsList;
        public void InitializeModel(string modelPath)
        {
            var option = new SessionOptions();
            option.GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL;
            option.ExecutionMode = ExecutionMode.ORT_PARALLEL;
            sess = new InferenceSession(modelPath, option);
        }

        public void Process(string inputPath, double opacity = 0.6)
        {
            resultsList = new List<AnomalyResults>();
            var image = Cv2.ImRead(inputPath);
            Bitmap hmResult;

            try
            {
                // Setup inputs and outputs
                var inputMeta = sess.InputMetadata;
                var inputName = inputMeta.Keys.ToArray()[0];
                int[] dim = inputMeta[inputName].Dimensions;
                var customMeta = sess.ModelMetadata.CustomMetadataMap;
                string lab = customMeta.ToArray()[0].Value;
                var lab_dict = JsonConvert.DeserializeObject<Dictionary<string, string>>(lab);
                string th = customMeta.ToArray()[1].Value;
                var th_dict = JsonConvert.DeserializeObject<Dictionary<string, string>>(th);
                string modelType = customMeta.ToArray()[2].Value;

                OpenCvSharp.Size imgSize = new OpenCvSharp.Size(dim[3], dim[2]);

                var data = DataPreprocessing(image);
                Mat imageFloat = data.Resize(imgSize);
                var input = new DenseTensor<float>(MatToList(imageFloat), new[] { dim[0], dim[1], dim[2], dim[3] });

                var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, input) };

                using (var results = sess.Run(inputs))
                {
                    var resultsArray = results.ToArray();
                    var map_value = resultsArray[0].AsEnumerable<float>().ToArray();
                    var map_dim = resultsArray[0].AsTensor<float>().Dimensions.ToArray();
                    var score_value = resultsArray[1].AsEnumerable<float>().ToArray();
                    var score_dim = resultsArray[1].AsTensor<float>().Dimensions.ToArray();

                    float adaptive_threshold = float.Parse(th_dict["image_threshold"]);
                    float pix_threshold = float.Parse(th_dict["pixel_threshold"]);
                    float min = float.Parse(th_dict["min"]);
                    float max = float.Parse(th_dict["max"]);

                    var normalizeMap = map_value.Select(x => ((x - min) / (max - min)) * 255).ToArray();
                    var normalizeMap_ori = map_value.Select(x => ((x - min) / (max - min)) * 255).ToArray();

                    double image_confidence = ((score_value[0] - adaptive_threshold) / (max - min)) + 0.5;
                    double pixel_confidence = ((score_value[0] - pix_threshold) / (max - min)) + 0.5;

                    double confidence = ((score_value[0] - adaptive_threshold) / (max - min)) + 0.5;

                    // To bound the confident to be in 0-1. 
                    if (confidence > 1)
                    {
                        confidence = 1;
                    }
                    else if (confidence < 0)
                    {
                        confidence = 0;
                    }

                    var heatMap = GetHeatmap(map_value, map_dim, normalizeMap_ori);
                    Cv2.ApplyColorMap(heatMap, heatMap, ColormapTypes.Jet);
                    OpenCvSharp.Size originalSize = new OpenCvSharp.Size(image.Width, image.Height);
                    Mat heatMap_resize = heatMap.Resize(originalSize);
                    Cv2.AddWeighted(image, opacity, heatMap_resize, 1 - opacity, 0, heatMap_resize);

                    using (var ms = heatMap_resize.ToMemoryStream())
                    {
                        hmResult = (Bitmap)Image.FromStream(ms);
                    }

                    resultsList.Add(new AnomalyResults((float)confidence * 100, hmResult));

                }
            }
            catch (Exception e)
            {
                Console.WriteLine("Model not initialized");
                Console.WriteLine(e.Message);
                using (var ms = image.ToMemoryStream())
                {
                    hmResult = (Bitmap)Image.FromStream(ms);
                }
            }
        }

        public static Mat GetHeatmap(float[] output_value, int[] output_dim, float[] HM_value)
        {
            Mat mat = new Mat(new OpenCvSharp.Size(output_dim[2], output_dim[3]), MatType.CV_8UC3);
            for (int batch = 0; batch < output_dim[0]; batch++)
            {
                for (int cls = 0; cls < output_dim[1]; cls++)
                {
                    for (int h = 0; h < output_dim[2]; h++)
                    {
                        for (int w = 0; w < output_dim[3]; w++)
                        {
                            int idx = (batch * output_dim[1] * output_dim[2] * output_dim[3]) + (cls * output_dim[2] * output_dim[3]) + (h * output_dim[3]) + w;

                            Vec3b pix = mat.At<Vec3b>(h, w);
                            pix = new Vec3b((byte)HM_value[idx], (byte)HM_value[idx], (byte)HM_value[idx]);
                            mat.Set<Vec3b>(h, w, pix);
                        }
                    }
                }
            }
            return mat;
        }
        public static Mat GetResultMask(float[] output_value, int[] output_dim, float output2_value)
        {
            Mat mat = new Mat(new OpenCvSharp.Size(output_dim[2], output_dim[3]), MatType.CV_8UC3);
            for (int batch = 0; batch < output_dim[0]; batch++)
            {
                for (int cls = 0; cls < output_dim[1]; cls++)
                {
                    for (int h = 0; h < output_dim[2]; h++)
                    {
                        for (int w = 0; w < output_dim[3]; w++)
                        {
                            int idx = (batch * output_dim[1] * output_dim[2] * output_dim[3]) + (cls * output_dim[2] * output_dim[3]) + (h * output_dim[3]) + w;

                            Vec3b pix = mat.At<Vec3b>(h, w);
                            if (output_value[idx] < output2_value)
                            {
                                pix = new Vec3b(0, 0, 0);
                            }
                            else
                            {
                                pix = new Vec3b(255, 255, 255);
                            }
                            mat.Set<Vec3b>(h, w, pix);
                        }
                    }
                }
            }
            return mat;
        }
        private Mat DataPreprocessing(Mat image)
        {
            Mat data = Mat.Zeros(image.Size(), MatType.CV_32FC3);
            using (var rgbImage = new Mat())
            {
                Cv2.CvtColor(image, rgbImage, ColorConversionCodes.BGR2RGB);
                rgbImage.ConvertTo(data, MatType.CV_32FC3, (float)(1 / 255.0));
                var channelData = Cv2.Split(data);
                channelData[0] = (channelData[0] - 0.485) / 0.229;
                channelData[1] = (channelData[1] - 0.456) / 0.224;
                channelData[2] = (channelData[2] - 0.406) / 0.225;
                Cv2.Merge(channelData, data);
            }
            return data;
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

        public void Dispose()
        {
            sess?.Dispose();
            sess = null;
        }

    }
}
