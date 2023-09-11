using Microsoft.AI.MachineLearning;
using System.Threading.Tasks;

namespace MachineLearning.Helpers
{
    internal class StringLengthHelper
    {
        public Task<TensorInt64Bit> GetStringTensor(string inputString)
        {
            int length = inputString.Length;
            var tensorData = new long[] { length };

            return Task.FromResult(TensorInt64Bit.CreateFromArray(new long[] { 1, 1 }, tensorData));
        }

        public Task<bool> IsStringLengthValid(TensorInt64Bit result)
        {
            // Get the first value from the tensor's vector view and convert it to a bool
            var resultValue = result.GetAsVectorView()[0];
            bool isValid = resultValue != 0;  // Assuming that non-zero means "valid"

            return Task.FromResult(isValid);
        }
    }
}
