using System;
using System.Collections.Generic;
using System.Linq;
using FluentAssert;
using NUnit.Framework;

namespace UnsupervisedCompetativeNeuralNet.Tests
{
    [TestFixture]
    public class UnsupervisedCompetativeNeuralNetTests
    {
        private const int MaxTries = 100;
        private const float LearningRate = 0.2f;
        private const float BiasLearningRate = 0.01f;

        [Test]
        public void ToyDataSet()
        {
            var dataset = new List<float[]>
            {
                new [] {1.0f, 0.9f, 0.8f},
                new [] {0.0f, 0.1f, 0.2f},
                new [] {0.2f, 0.1f, 0.1f},
                new [] {0.9f, 0.9f, 0.8f}
            };
            var expectedCluster = new List<int[]>
            {
                new [] {0, 3},
                new [] {1, 2}
            };

            TrainOnDataSet(dataset, expectedCluster).ShouldBeTrue();
        }

        [Test]
        public void OnlyASingleValueMatters()
        {
            var dataset = new List<float[]>
            {
                new [] {1.0f, 0.9f, 1.0f},
                new [] {1.0f, 1.0f, 0.0f},
                new [] {1.0f, 0.9f, 0.0f},
                new [] {0.9f, 1.0f, 1.0f}
            };
            var expectedCluster = new List<int[]>
            {
                new [] {0, 3},
                new [] {1, 2}
            };

            TrainOnDataSet(dataset, expectedCluster).ShouldBeTrue();
        }

        [Test]
        public void ThreeClasses()
        {
            var dataset = new List<float[]>
            {
                new[] {0.2f, 0.9f},
                new[] {0.2f, 0.8f},
                new[] {0.9f, 0.9f},
                new[] {0.8f, 0.9f},
                new[] {0.5f, 0.1f},
                new[] {0.3f, 0.0f}
            };
            var expectedCluster = new List<int[]>
            {
                new[] {0, 1},
                new[] {2, 3},
                new[] {4, 5}
            };

            TrainOnDataSet(dataset, expectedCluster).ShouldBeTrue();
        }

        private static bool TrainOnDataSet(IList<float[]> dataSet, IList<int[]> expectedClusters)
        {
            var unsupervisedNueralNet = new UnsupervisedCompetativeNeuralNet(dataSet.First().Length,
                expectedClusters.Count(), LearningRate, BiasLearningRate);
            for (var i = 0; i < MaxTries; i++)
            {
                unsupervisedNueralNet.Train(dataSet);

                if (ValidateClusters(dataSet, expectedClusters, unsupervisedNueralNet))
                    return true;
            }

            foreach (var item in dataSet)
            {
                var cluster = unsupervisedNueralNet.GetCluster(item);
                Console.WriteLine("{0} = {1}", item, cluster);
            }

            return false;
        }

        private static bool ValidateClusters(IList<float[]> dataSet, IEnumerable<int[]> expectedClusters, UnsupervisedCompetativeNeuralNet neuralNet)
        {
            return expectedClusters
                .All(expectedCluster =>
                        expectedCluster.Select(x => neuralNet.GetCluster(dataSet[x])).GroupBy(x => x).Count() == 1);
        }
    }
}
