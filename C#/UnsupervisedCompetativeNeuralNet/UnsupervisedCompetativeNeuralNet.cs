using System;
using System.Collections.Generic;

namespace UnsupervisedCompetativeNeuralNet
{
    public class UnsupervisedCompetativeNeuralNet
    {
        private readonly float[] biases;
        private readonly float[][] connections;
        private readonly float[] hiddenNodes;
        private readonly int hiddenNodesCount;
        private readonly int inputNodesCount;

        public UnsupervisedCompetativeNeuralNet(int inputNodesCount, int hiddenNodesCount, float learningRate,
            float biasLearningRate)
        {
            if (hiddenNodesCount < 1)
                throw new ArgumentException("must be greater than 1", "hiddenNodesCount");
            if (inputNodesCount < 0)
                throw new ArgumentException("must be greater than 0", "inputNodesCount");
            if (learningRate < 0)
                throw new ArgumentException("must be greater than 0", "learningRate");
            if (learningRate > 1.0)
                throw new ArgumentException("must be less than 1", "learningRate");

            this.inputNodesCount = inputNodesCount;
            this.hiddenNodesCount = hiddenNodesCount;
            LearningRate = learningRate;
            BiasLearningRate = biasLearningRate;
            biases = new float[this.hiddenNodesCount];
            connections = new float[this.inputNodesCount][];

            var random = new Random();
            for (var i = 0; i < this.inputNodesCount; i++)
            {
                connections[i] = new float[this.hiddenNodesCount];
                for (var j = 0; j < this.hiddenNodesCount; j++)
                    connections[i][j] = (float) random.NextDouble();
            }
            hiddenNodes = new float[this.hiddenNodesCount];
        }

        public float LearningRate { get; set; }
        public float BiasLearningRate { get; set; }

        public int GetCluster(float[] data)
        {
            FeedForward(data);
            return GetIndexOfMax(hiddenNodes);
        }

        public int Train(float[] data)
        {
            FeedForward(data);
            var winningCluster = GetIndexOfMax(hiddenNodes);
            for (var i = 0; i < inputNodesCount; i++)
                connections[i][winningCluster] += LearningRate*(data[i] - connections[i][winningCluster]);

            biases[winningCluster] -= BiasLearningRate;

            return winningCluster;
        }

        public void Train(IEnumerable<float[]> dataSet)
        {
            foreach (var data in dataSet)
                Train(data);
        }

        private static int GetIndexOfMax(float[] array)
        {
            var maxIndex = 0;
            var maxValue = array[0];
            for (var i = 1; i < array.Length; i++)
            {
                if (array[i] > maxValue)
                {
                    maxValue = array[i];
                    maxIndex = i;
                }
            }

            return maxIndex;
        }

        private void FeedForward(float[] data)
        {
            for (var i = 0; i < hiddenNodesCount; i++)
            {
                var activation = biases[i];
                for (var j = 0; j < inputNodesCount; j++)
                    activation += data[j] - connections[j][i];
                hiddenNodes[i] = activation;
            }
        }
    }
}