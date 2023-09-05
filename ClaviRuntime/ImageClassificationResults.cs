using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ClaviRuntime
{
    public class ImageClassificationResults
    {
        private string name;
        private float score;
        public ImageClassificationResults(string name, float score)
        {
            this.name = name;
            this.score = score;
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
