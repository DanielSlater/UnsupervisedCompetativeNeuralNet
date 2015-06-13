import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

/**
 * Created by Daniel on 07/06/2015.
 */
public class UnsupervisedCompetativeNeuralNetTests {
    private static final int MaxTries = 100;
    private static final float LearningRate = 0.2f;
    private static final float BiasLearningRate = 0.01f;

    @Test
    public void ToyDataSet()
    {
        float[][] dataset = new float[][]
        {
            {1.0f, 0.9f, 0.8f},
            {0.0f, 0.1f, 0.2f},
            {0.2f, 0.1f, 0.1f},
            {0.9f, 0.9f, 0.8f}
        };
        int[][] expectedCluster = new int[][]
        {
            {0, 3},
            {1, 2}
        };

        Assert.assertTrue(trainOnDataSet(Arrays.asList(dataset), expectedCluster));
    }

    @Test
    public void OnlyASingleValueMatters()
    {
        float[][] dataset = new float[][]
        {
            {1.0f, 0.9f, 1.0f},
            {1.0f, 1.0f, 0.0f},
            {1.0f, 0.9f, 0.0f},
            {0.9f, 1.0f, 1.0f}
        };
        int[][] expectedCluster = new int[][]
        {
            {0, 3},
            {1, 2}
        };

        Assert.assertTrue(trainOnDataSet(Arrays.asList(dataset), expectedCluster));
    }

    @Test
    public void ThreeClasses()
    {
        float[][] dataset = new float[][]
        {
            {0.2f, 0.9f},
            {0.2f, 0.8f},
            {0.9f, 0.9f},
            {0.8f, 0.9f},
            {0.5f, 0.1f},
            {0.3f, 0.0f}
        };
        int[][] expectedCluster = new int[][]
        {
            {0, 1},
            {2, 3},
            {4, 5}
        };

        Assert.assertTrue(trainOnDataSet(Arrays.asList(dataset), expectedCluster));
    }

    private static boolean trainOnDataSet(List<float[]> dataSet, int[][] expectedClusters)
    {
        UnsupervisedCompetativeNeuralNet unsupervisedNeuralNet = new UnsupervisedCompetativeNeuralNet(dataSet.get(0).length,
                expectedClusters.length, LearningRate, BiasLearningRate);
        for (int i = 0; i < MaxTries; i++)
        {
            unsupervisedNeuralNet.train(dataSet);

            if (validateClusters(dataSet, expectedClusters, unsupervisedNeuralNet))
                return true;
        }
        for (float[] item : dataSet) {
            int cluster = unsupervisedNeuralNet.getCluster(item);
            System.err.printf("{0} = {1}", item, cluster);
        }

        return false;
    }

    private static boolean validateClusters(List<float[]> dataSet, int[][] expectedClusters,
                                            UnsupervisedCompetativeNeuralNet neuralNet)
    {
        for (int[] expectedCluster : expectedClusters) {
            int clusterNumber = -1;

            for (int i : expectedCluster) {
                int currentCluster = neuralNet.getCluster(dataSet.get(i));
                if(clusterNumber == -1)
                    clusterNumber = currentCluster;
                else if (clusterNumber != currentCluster)
                    return false;
            }
        }

        return true;
    }
}
