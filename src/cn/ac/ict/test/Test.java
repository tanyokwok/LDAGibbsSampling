package cn.ac.ict.test;

import cn.ac.ict.lda.LDAGibbsInference;
import cn.ac.ict.lda.LDAGibbsSampler;
import cn.ac.ict.lda.TopicWordSelector;
import cn.ac.ict.text.Vocabulary;
import cn.ac.ict.text.inference.DocumentLoader;
import cn.ac.ict.text.train.DocuVocaLoader;

public class Test {
	public static void train(){
		String test_dir = "./src/data/train";
		DocuVocaLoader loader = new DocuVocaLoader();
		loader.loadDocuments(test_dir);
		// words in documents 
        int[][] documents = loader.getCorpus();
        String voca[] = loader.getVocabulary();
        int M = documents.length;
        int K = 20;
        int V = voca.length;
        LDAGibbsSampler lda = new LDAGibbsSampler(documents,V);
        lda.config(K,2000,10000,10);
        
        lda.gibbsSample(2, .5);
        //输出模型参数，论文中式 （81）与（82）  
        double[][] theta = lda.normalizeTheta();
        double[][] phi = lda.normalizePhi();  
        
        TopicWordSelector selector = new TopicWordSelector(10);
        for( int k = 0; k < phi.length; ++ k ){
        	int topic_words[] = selector.getTopicWord(phi[k]);
        	for( int i : topic_words )
        		System.out.print( voca[i] +" ");
        	System.out.println();
        }
        
        lda.dumpPhi("./src/data/phi", phi);
        Vocabulary.dumpVocabulary("./src/data/vocabulary", voca);
	}
	public static void main(String args[]){
		int K = 20;
        Vocabulary vocabulary = new Vocabulary();
        
        vocabulary.loadVocabulary("./src/data/vocabulary");
        int V = vocabulary.vocabulary.length;
        int doc[] = DocumentLoader.loadDocument("./src/data/test/sports000061.txt", vocabulary);
        LDAGibbsInference ldainfer = new LDAGibbsInference(doc,V);
        double phi[][] = LDAGibbsInference.loadPhi("./src/data/phi");
        
        TopicWordSelector selector = new TopicWordSelector(30);
        for( int k = 0; k < phi.length; ++ k ){
        	int topic_words[] = selector.getTopicWord(phi[k]);
        	for( int i : topic_words )
        		System.out.print( vocabulary.vocabulary[i] +" ");
        	System.out.println();
        }
        
        ldainfer.config(K,2000,10000,10);
        ldainfer.gibbsSample(2, phi);
        double pzd[] = ldainfer.normalizeTheta();
        for( double d: pzd )
        	System.out.println(d);
	}

}
