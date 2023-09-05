using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ClaviRuntime
{
    public class AnomalyResults
    {
        private float score;
        private Bitmap heatmap;
        public AnomalyResults(float score, Bitmap heatmap)
        {
            this.score = score;
            this.heatmap = heatmap;
        }
        public float Score
        {
            get { return score; }
            set { score = value; }
        }
        public Bitmap Heatmap
        {
            get { return heatmap; }
            set { heatmap = value; }
        }
    }
}
