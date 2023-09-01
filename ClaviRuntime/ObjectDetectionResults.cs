using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ClaviRuntime
{
    public class ObjectDetectionResults
    {
        private int index;
        private string label;
        private float score;
        private float[] box;

        public ObjectDetectionResults(int index, string label, float score, float[] box)
        {
            this.index = index;
            this.label = label;
            this.score = score;
            this.box = box;
        }

        public int Index
        {
            get { return index; }
            set { index = value; }
        }
        public string Label
        {
            get { return label; }
            set { label = value; }
        }
        public float Score
        {
            get { return score; }
            set { score = value; }
        }
        public float[] Box
        {
            get { return box; }
            set { box = value; }
        }
    }
}
