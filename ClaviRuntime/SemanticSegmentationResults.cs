using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ClaviRuntime
{
    public class SemanticSegmentationResults
    {
        private Bitmap mask;
        public SemanticSegmentationResults(Bitmap mask)
        {
            this.mask = mask;
        }
        public Bitmap Mask
        {
            get { return mask; }
            set { mask = value; }
        }
    }
}
