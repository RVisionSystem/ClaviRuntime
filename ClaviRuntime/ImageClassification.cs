using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using Newtonsoft.Json;
using OpenCvSharp;

namespace ClaviRuntime
{
    public class ImageClassification : IDisposable
    {
        private InferenceSession? sess;
        public List<ImageClassificationResults>? resultsList;

        public void InitializeModel(string modelPath)
        {
            var option = new SessionOptions();
            option.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            option.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            sess = new InferenceSession(modelPath, option);
        }

        public void Process(string imagePath, float threshold = 0.7F)
        {
            resultsList = new List<ImageClassificationResults>();
            Mat image = Cv2.ImRead(imagePath);
            try
            {
                // Setup inputs and outputs
                var inputMeta = sess.InputMetadata;
                var inputName = inputMeta.Keys.ToArray()[0];
                var meta = inputMeta[inputName];
                int[] dims = meta.Dimensions;
                var customMeta = sess.ModelMetadata.CustomMetadataMap;
                string lab = customMeta.ToArray()[0].Value;
                var lab_dict = JsonConvert.DeserializeObject<Dictionary<string, string>>(lab);

                Size imgSize = new Size(dims[3], dims[2]);

                Mat imageFloat = image.Resize(imgSize);
                imageFloat.ConvertTo(imageFloat, MatType.CV_32FC1);
                var input = new DenseTensor<float>(MatToList(imageFloat), new[] { dims[0], dims[1], dims[2], dims[3] });

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(inputName, input)
                };
                using (var results = sess.Run(inputs))
                {
                    var resultsArray = results.ToArray();
                    var pred_value = resultsArray[0].AsEnumerable<float>().ToArray();
                    var pred_dim = resultsArray[0].AsTensor<float>().Dimensions.ToArray();
                    float maxValue = pred_value.Max();
                    int maxIndex = pred_value.ToList().IndexOf(maxValue);

                    resultsList.Add(new ImageClassificationResults(lab_dict[maxIndex.ToString()], pred_value[maxIndex]));
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("Model not initialized");
                Console.WriteLine(e.Message);
            }
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
