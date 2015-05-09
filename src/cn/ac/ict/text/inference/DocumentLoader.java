package cn.ac.ict.text.inference;

import java.io.File;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.log4j.Logger;

import cn.ac.ict.text.StopWordSingleton;
import cn.ac.ict.text.TextDocuReaderSingleton;
import cn.ac.ict.text.Vocabulary;

public class DocumentLoader {
	
	public static int [] loadDocument(String filename,Vocabulary voca){
		
		TextDocuReaderSingleton singleton = TextDocuReaderSingleton.sigleton;
	
		List<String> word_list = singleton.readDocument(new File(filename), "\\|\\|");
		word_list = StopWordSingleton.singleton.filteStopWords(word_list);
		
		int doc[] = new int[word_list.size()];
		int idx = 0;
		for( String word: word_list ){
			Integer wid = voca.wordId.get(word);
			if( wid == null ) doc[idx++] = 0;
			else doc[idx++] = wid;
		}
		
		return doc;
	}
	

	public static void main(String args[]){
		double small = Double.parseDouble("1.0E-12");
		System.out.println(small);
		System.out.printf("%.13f",small);
//		String test_dir = "./src/data";
//		DocumentLoader loader = new DocumentLoader();
//		loader.loadDocuments(test_dir);
//		File file_list[] = loader.getFileList(test_dir);
//		for( File f: file_list ){
//			System.out.println(f.getAbsolutePath());
//		}
	}
	
}
