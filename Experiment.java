
public class Experiment{

    private final int dataSize = 1024 * 1024;

    private byte[][] data;

    private int[] dataBlockSizes;

    private float[][] pattern;

    public Experiment(int[] dataBlockSizes){
        this.initializeData();

        this.dataBlockSizes = dataBlockSizes;

        this.initializePattern();
    }

    public Experiment(){
        this.initializeData();

        int[] blocks = {8, 16, 24, 32, 40, 48, 56, 64};
        this.dataBlockSizes = blocks;

        this.initializePattern();
    }

    public Experiment(int patternSizeScale){
        this.initializeData();

        int[] blocks = {8, 16, 24, 32, 40, 48, 56, 64};
        this.dataBlockSizes = blocks;

        this.initializePattern(patternSizeScale);
    }

    private void initializeData(){
        this.data = new byte[this.dataSize][];
    }

    private void initializePattern(){
        /**
         * Pattern format:
         * pattern[index][j]
         * 
         * Where j: 0 is the blocksize
         * j: 1: mode: 0 for deallocate, 1 for read/write a random index, 2 for allocate at empty slot
         * j: 2: probability: probability for that action to happen
         */

        pattern = new float[(this.dataBlockSizes.length * 3)][];

        int blockSizeIndex = 0;

        for(int i = 0; i < pattern.length; i+=3){
            float[] entry = new float[3];
            entry[0] = this.dataBlockSizes[blockSizeIndex];
            entry[1] = 0;
            entry[2] = (float)Math.random();
            pattern[i] = entry;

            entry = new float[3];
            entry[0] = this.dataBlockSizes[blockSizeIndex];
            entry[1] = 1;
            entry[2] = (float)Math.random();
            pattern[i + 1] = entry;

            entry = new float[3];
            entry[0] = this.dataBlockSizes[blockSizeIndex];
            entry[1] = 2;
            entry[2] = (float)Math.random();
            pattern[i + 2] = entry;

            blockSizeIndex++;
        }

        //Do a simple shuffle
        for(int i = 0; i < pattern.length; i++){
            int j = (int)(Math.random() * pattern.length);
            int k = (int)(Math.random() * pattern.length);

            //Swap those indicies
            float[] temp = pattern[j];
            pattern[j] = pattern[k];
            pattern[k] = temp;
        }
    }

    private void initializePattern(int patternSizeScale){
        /**
         * Pattern format:
         * pattern[index][j]
         * 
         * Where j: 0 is the blocksize
         * j: 1: mode: 0 for deallocate, 1 for read/write a random index, 2 for allocate at empty slot
         * j: 2: probability: probability for that action to happen
         */

        pattern = new float[(this.dataBlockSizes.length * 3) * patternSizeScale][];

        int blockSizeIndex = 0;

        for(int i = 0; i < pattern.length; i+=3){
            float[] entry = new float[3];
            entry[0] = this.dataBlockSizes[blockSizeIndex % this.dataBlockSizes.length];
            entry[1] = 0;
            entry[2] = (float)Math.random();
            pattern[i] = entry;

            entry = new float[3];
            entry[0] = this.dataBlockSizes[blockSizeIndex % this.dataBlockSizes.length];
            entry[1] = 1;
            entry[2] = (float)Math.random();
            pattern[i + 1] = entry;

            entry = new float[3];
            entry[0] = this.dataBlockSizes[blockSizeIndex % this.dataBlockSizes.length];
            entry[1] = 2;
            entry[2] = (float)Math.random();
            pattern[i + 2] = entry;

            blockSizeIndex++;
        }

        //Do a simple shuffle
        for(int i = 0; i < pattern.length; i++){
            int j = (int)(Math.random() * pattern.length);
            int k = (int)(Math.random() * pattern.length);

            //Swap those indicies
            float[] temp = pattern[j];
            pattern[j] = pattern[k];
            pattern[k] = temp;
        }
    }


    


    private void iterate(float[] entry){

        int blockSize = (int)entry[0];
        int mode = (int)entry[1];
        float prob = entry[2];

        for(int i = 0; i < this.data.length; i++){
            if(mode == 0 && Math.random() < prob){
                //deallocate
                this.data[i] = null;
            } else if(mode == 1 && Math.random() < prob){
                //read/write if an entry exists
                if(this.data[i] != null){
                    //Get a random index, read and write a value to it.

                    int randomIndex = (int)(Math.random() * this.data[i].length);
                    byte b = this.data[i][randomIndex];
                    this.data[i][randomIndex] = 'c';
                }
                
            } else {
                //allocate if no block is here
                if(this.data[i] == null && Math.random() < prob){
                    this.data[i] = new byte[blockSize];
                }
            }
        }
    }

    public float performExperimentDefault(int iterations){

        long startTime = System.currentTimeMillis() / 1000;

        for(int i = 0; i < iterations; i++){
            iterate(this.pattern[i % this.pattern.length]);
        }

        long endTime = System.currentTimeMillis() / 1000;

        return (int)(endTime - startTime) / ((float)iterations);
    }


    public float performExperimentAgent(int iterations, float probStop, int hiddenLayerSize, float baseline){
        long startTime = System.currentTimeMillis() / 1000;

        GCAgent agent = new GCAgent(hiddenLayerSize, hiddenLayerSize, 0.0001f, 0.001f, 0.00001f, 0.9999f, 1000);
        
        agent.setBaseline(baseline);

        for(int i = 0; i < iterations; i++){
            iterate(this.pattern[i % this.pattern.length]);

            agent.invoke();

            if(agent.getAvgProb() >= probStop){
                return agent.getAvgTimeInterval();
            }
            
        }

        long endTime = System.currentTimeMillis() / 1000;

        return (int)(endTime - startTime) / ((float)iterations);
    }


    public int manualGC(int iterations, boolean useManualGC){
        long startTime = System.currentTimeMillis() / 1000;

        byte[][] data = new byte[3000000][];
        int bigAllocations = 32;
        int smallAllocations = 4;

        /**
         * Steps:
         * 0: Remove many bigger allocations
         * 1: create many small allocations
         * 2: Remove half of the small allocations
         * 3: Create many bigger allocations
         */

        for(int i = 0; i < iterations; i++){
            for(int j = 0; j < data.length; j++){
                if(i % 4 == 0 && data[j] != null && data[j].length == bigAllocations){
                    double r = Math.random();
                    if(r > 0.1){
                        data[j] = null;
                    }
                } else if(i % 4 == 1 && data[j] == null){
                    double r = Math.random();
                    if(r > 0.1){
                        data[j] = new byte[smallAllocations];
                    }
                } else if(i % 4 == 2 && data[j] != null && data[j].length == smallAllocations){
                    double r = Math.random();
                    if(r > 0.5){
                        data[j] = null;
                    }
                } else if(i % 4 == 3 && data[j] == null){
                    double r = Math.random();
                    if(r > 0.1){
                        data[j] = new byte[bigAllocations];
                    }
                }
            }

            if(i % 4 == 2 && useManualGC){
                System.gc();
            }
        }

        long endTime = System.currentTimeMillis() / 1000;

        return (int)(endTime - startTime);
    }


    public static void main(String[] args){
        Experiment exp = new Experiment(100);

        float defaultTime = exp.performExperimentDefault(10 * 1000);
        System.out.println("Default time: " + defaultTime);

        for(int h = 60; h <= 80; h += 10){

            float agentTime = exp.performExperimentAgent(1000 * 1000 * 30, 0.98f, h, defaultTime);

            System.out.println("Hidden layer size: " + h);
            System.out.println("Performance Ratio: " + ((float)defaultTime) / agentTime);
        }

    }
}