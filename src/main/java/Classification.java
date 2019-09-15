import org.apache.log4j.BasicConfigurator;
import org.datavec.api.io.converters.SelfWritableConverter;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.slf4j.Logger;

import java.io.FileNotFoundException;
import java.io.IOException;

public class Classification {
    private final static Logger LOGGER = org.slf4j.LoggerFactory.getLogger(Classification.class);

    public static void main(String[] args) throws IOException, InterruptedException {
        int batchSize = 1024;
        int labelIndex = 6;
        int numPossibleLabels = 2;
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
        CSVRecordReader trainReader = new CSVRecordReader(1, ",");
        trainReader.initialize(new FileSplit(new ClassPathResource("data/classification/train.csv")
                .getFile()));
        RecordReaderDataSetIterator trainIter = new RecordReaderDataSetIterator(trainReader, new SelfWritableConverter(), batchSize, labelIndex, numPossibleLabels, false);
        normalizer.fit(trainIter);
        trainIter.setPreProcessor(normalizer);

    }
}
