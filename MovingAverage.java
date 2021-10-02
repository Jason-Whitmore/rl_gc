import java.util.concurrent.ArrayBlockingQueue;

public class MovingAverage{

    private int maxSize;

    private int currentSize;

    private float sum;

    private ArrayBlockingQueue<Float> queue;

    public MovingAverage(int samples){

        this.maxSize = samples;
        this.currentSize = 0;
        this.sum = 0;
        this.queue = new ArrayBlockingQueue<Float>(samples);
    }

    public void addSample(float sample){

        if(this.currentSize == this.maxSize){
            //remove oldest sample, subtract from sum
            float oldestSample = this.queue.poll();
            this.sum -= oldestSample;

            //Add new sample
            this.queue.offer(sample);
            this.sum += sample;
        } else {
            //Add new sample, increment size
            this.queue.offer(sample);
            this.sum += sample;
            currentSize++;
        }
    }

    public void reset(){
        this.currentSize = 0;
        this.sum = 0;
        this.queue = new ArrayBlockingQueue<Float>(this.maxSize);
    }

    public float getMean(){
        if(this.currentSize == 0){
            return 0f;
        }

        return this.sum / this.currentSize;
    }

    public float getNumSamples(){
        return this.currentSize;
    }


}