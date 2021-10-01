import java.io.FileWriter;
import java.io.IOException;

public class CSVWriter {

    private String dataString;

    private String filePath;

    private int numCols;

    public CSVWriter(String filePath, String[] columnHeaders){
        this.filePath = filePath;

        this.numCols = columnHeaders.length;

        this.dataString = "";

        addRow(columnHeaders);
    }

    public void addRow(String[] newRow){
        for(int i = 0; i < newRow.length - 1; i++){
            dataString += newRow[i] + ", ";
        }

        dataString += newRow[newRow.length - 1] + "\n";
    }

    public void writeToFile(){

        try{
            FileWriter writer = new FileWriter(this.filePath);
            writer.write(this.dataString);
            writer.close();
        } catch(IOException exception){
            System.err.println("IO exception occured in writeToFile().");
            System.err.println(exception);
        }

    }


}