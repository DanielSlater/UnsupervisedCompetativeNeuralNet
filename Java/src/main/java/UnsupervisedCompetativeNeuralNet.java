import java.util.Random;

/**
 * Created by Daniel on 19/05/2015.
 */
public class UnsupervisedCompetativeNeuralNet {
    private final int inputNodesCounts;
    private final int hiddenNodesCount;
    private float learningRate;
    private float biasLearningRate;
    private final float[][] connections;
    private final float[] biases;
    private final float[] hiddenNodes;

    public UnsupervisedCompetativeNeuralNet(int inputNodesCounts, int hiddenNodesCount,
                                            float learningRate, float biasLearningRate){
        this.inputNodesCounts = inputNodesCounts;
        this.hiddenNodesCount = hiddenNodesCount;
        this.learningRate = learningRate;
        this.biasLearningRate = biasLearningRate;
        biases = new float[this.hiddenNodesCount];
        hiddenNodes = new float[this.hiddenNodesCount];

        Random random = new Random();
        connections = new float[this.inputNodesCounts][];
        for (int i = 0; i < inputNodesCounts; i++){
            connections[i] = new float[this.hiddenNodesCount];
            for (int h =0; h < hiddenNodesCount; h++){
                connections[i][h] = random.nextFloat();
            }
        }
    }

    private void FeedForward(float[] data){
        for(int h=0; h<hiddenNodesCount;h++){
            float activation = biases[h];
            for(int i=0; i<inputNodesCounts; i++){
                activation += data[i]*connections[i][h];
            }

            this.hiddenNodes[h] = activation;
        }
    }

    public int train(float[] data){
        FeedForward(data);
        int winningNode = getIndexOfMax(hiddenNodes);

        for(int i = 0; i< inputNodesCounts; i++){
            connections[i][winningNode] += learningRate*(data[i]-connections[i][winningNode]);
        }

        return winningNode;
    }

    public void train(Iterable<float[]> dataSet){
        for (float[] data : dataSet) {
            train(data);
        }
    }

    public int getCluster(float[] data){
        FeedForward(data);
        return getIndexOfMax(this.hiddenNodes);
    }

    private static int getIndexOfMax(float[] items){
        float maxValue = items[0];
        int maxIndex = 0;
        for(int i=1;i<items.length;i++){
            if(items[i] > maxValue){
                maxValue = items[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    public float getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    public float getBiasLearningRate() {
        return biasLearningRate;
    }

    public void setBiasLearningRate(float biasLearningRate) {
        this.biasLearningRate = biasLearningRate;
    }
}
