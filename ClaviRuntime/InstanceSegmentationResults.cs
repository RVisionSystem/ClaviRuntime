using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json;
using OpenCvSharp.Dnn;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ClaviRuntime
{
    public class InstanceSegmentationResults
    {
        private string id;
        private string name;
        private float score;

        public InstanceSegmentationResults(string id, string name, float score)
        {
            this.id = id;
            this.name = name;
            this.score = score;
        }
        public string Id
        {
            get { return id; }
            set { id = value; }
        }
        public string Name
        {
            get { return name; }
            set { name = value; }
        }
        public float Score
        {
            get { return score; }
            set { score = value; }
        }
    }
}
