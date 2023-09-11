using Microsoft.AI.MachineLearning;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MachineLearning.Helpers
{
    internal class IpRangeHelper
    {
        public Task<TensorInt64Bit> GetStringTensor(string inputString)
        {
            var tensorData = ConvertIpRangeToNumericList(inputString);
            return Task.FromResult(TensorInt64Bit.CreateFromArray(new long[] { 1, 15 }, tensorData));
        }

        public Task<bool> IsStringLengthValid(TensorInt64Bit result)
        {
            // Get the first value from the tensor's vector view and convert it to a bool
            var resultValue = result.GetAsVectorView()[0];
            bool isValid = resultValue != 0;  // Assuming that non-zero means "valid"

            return Task.FromResult(isValid);
        }

        static long[] ConvertIpRangeToNumericList(string inputStr)
        {
            List<int> output = new List<int>();
            inputStr = inputStr.Replace(" ", "");

            string[] ipRanges = inputStr.Split(',');
            foreach (var ipRange in ipRanges)
            {
                string[] octets = ipRange.Split('.');
                int missingOctets = 4 - octets.Length;

                foreach (var octet in octets)
                {
                    output.AddRange(ParseOctet(octet));
                }

                // Append -1 for each missing octet
                output.AddRange(Enumerable.Repeat(-1, missingOctets));
            }

            for (int i = 0; i < 15; ++i)
            {
                if(i >= output.Count)
                {
                    output.Add(0);
                }
            }

            return output.Select(x => (long)x).ToArray();
        }

        static List<int> ParseOctet(string octet)
        {
            List<int> result = new List<int>();
            if (int.TryParse(octet, out int value))
            {
                result.Add(value);
            }
            else if (octet.Contains("-"))
            {
                string[] parts = octet.Split('-');
                if (parts.Length == 2 && int.TryParse(parts[0], out int start) && int.TryParse(parts[1], out int end))
                {
                    result.Add(start);
                    result.Add(end);
                }
            }

            if (!result.Any())
            {
                result.Add(-1);
            }

            return result;
        }
    }
}
