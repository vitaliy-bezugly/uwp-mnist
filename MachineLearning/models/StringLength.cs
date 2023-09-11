using Microsoft.AI.MachineLearning;
using System.Threading.Tasks;
using System;
using Windows.Storage.Streams;
using System.Linq;
using System.Collections.Generic;

namespace MachineLearning
{
    public sealed class StringLengthInput
    {
        public TensorInt64Bit Length;  // shape: [1, 1]
    }

    public sealed class StringLengthOutput
    {
        public TensorInt64Bit Result;  // shape: [1, 1]
    }

    public sealed class StringLengthModel
    {
        private LearningModel model;
        private LearningModelSession session;
        private LearningModelBinding binding;

        public static async Task<StringLengthModel> CreateFromStreamAsync(IRandomAccessStreamReference stream)
        {
            var learningModel = new StringLengthModel();

            learningModel.model = await LearningModel.LoadFromStreamAsync(stream);
            learningModel.session = new LearningModelSession(learningModel.model);
            learningModel.binding = new LearningModelBinding(learningModel.session);

            return learningModel;
        }

        public async Task<StringLengthOutput> EvaluateAsync(StringLengthInput input)
        {
            binding.Bind("input", input.Length);
            var result = await session.EvaluateAsync(binding, "0");
            var output = new StringLengthOutput();

            var outputLabel = result.Outputs["output_label"] as TensorInt64Bit;
            if(outputLabel == null)
            {
                throw new Exception($"Unexpected model evaluation result output type");
            }

            output.Result = outputLabel;
            return output;
        }
    }
}
