package cn.ac.ict.lda;
import java.util.Random;

import org.apache.log4j.Logger;

import cn.ac.ict.time.TimeCounter;


/**
 * LDAGibbsSampler implements a algorithm with complexity of O( iteration*K*corpus size)
 * where K is the topic number, corpus size is the total word number of the corpus.
 * @author Administrator
 */
public class LDAGibbsTraining extends LDAGibbsSampler{
	private static Logger logger = Logger.getLogger(LDAGibbsTraining.class);
	public LDAGibbsTraining(int [][]documents,int V ){
		super(documents,V);
	}
	
	/**
	 * Sample a topic z_i from the full conditional distribution: p(z_i = j | 
     * z_-i, w) = (n_-i,j(w_i) + beta)/(n_-i,j(.) + W * beta) * (n_-i,j(d_i) + 
     * alpha)/(n_-i,.(d_i) + K * alpha) 
	 * @param m document
	 * @param n the n-th word in the document
	 * @return
	 */
	public int sampleFullConditional(int m,int n){
		//remove zi from the time count variables
		int z = topics[m][n];
		int w = docs[m][n];
		nzw[z][w] -- ;
		nzd[z][m] -- ;
		cnz[z] --;
		cnd[m] --;
		
		// do multinomial sampling via cumulative method:
		double [] p = new double[K];
		for( int k = 0; k < K; ++ k ){
			p[k] = (( nzw[k][w] + beta)/( cnz[k] + V*beta ))*((nzd[k][m] + alpha )/( cnd[m] + K*alpha)); 
		}
		
		for( int k = 1 ; k < K; ++ k ){
			p[k] += p[k-1];//cdf
		}
//		double r = Math.random()*p[K-1];
		double r = rand.nextDouble()*p[K-1];
		for( z = 0; z < K; ++ z ){
			if( r < p[z] ){
				break;
			}
		}
		//add new topic z_i to count variables
		nzw[z][w] ++;
		nzd[z][m] ++;
		cnz[z] ++;
		cnd[m] ++;
		
		return z;
	}
	
	/**
	 * Main method: 
	 * 1. Select initial state 
	 * 2. Repeat a large number of times: 
	 * 	(a). Select an element 
	 * 	(b)  Update conditional on other elements.
	 * If appropriate, output summary for each run. 
	 */
	public void gibbsSample(double alpha,double beta){
		this.alpha = alpha;
		this.beta = beta;
		TimeCounter.updateTimeStamp(this);
		initSampleTopics();
		doTopicCounting();
		for( int i = 0; i < ITERATION; ++ i ){
			
			for( int m = 0; m < D; ++ m ){
				for( int n = 0; n < docs[m].length; ++ n){
					int topic = sampleFullConditional(m,n);
					topics[m][n] = topic;
				}
			}
			if( i % TIME_GAP == 0 ){
				logger.info("Running Time:  " + TimeCounter.getTimeDiff(this));
			}
			if( ( i > this.BURN_IN ) && ( this.SAMPLE_GAP > 0 ) && ( i % this.SAMPLE_GAP  == 0)){
				addSampleStatistics();
			}
		}
		
		TimeCounter.removeTimeStamp(this);
	}
	
	/**
	 * When a new sample is sampled add it to statistics variables.
	 */
	public void addSampleStatistics(){
		for( int k = 0; k < K; ++ k ){
			for( int m = 0; m < D; ++ m ){
				pzdsum[k][m] += (nzd[k][m] + alpha )/( cnd[m] + K*alpha);
			}
		}
		
		for( int w = 0; w < V; ++ w){
			for( int k = 0; k < K; ++ k){
				pwzsum[w][k] += ( nzw[k][w] + beta)/( cnz[k] + V*beta );
			}
		}
		numstat ++;
	}
	
	/**
	 * 
	 * @return a matrix of K*D represents theta
	 */
	public double [][] normalizeTheta(){
		double [][] theta = new double[K][D];
		for( int k = 0; k < K; ++ k ){
			for(int m = 0; m < D; ++ m ){
				theta[k][m] = pzdsum[k][m]/numstat;
			}
		}
		return theta;
	}
	
	/**
	 * normalize the probability of p(word|topic)
	 * @return a matrix of K*V represents phi, phi[ topic ][ word ] = p(word|topic)
	 */
	public double[][] normalizePhi(){
		double [][] phi = new double[K][V];
		for( int k = 0; k < K; ++ k ){
			for(int w = 0; w < V; ++ w ){
				phi[k][w] = pwzsum[w][k]/numstat;
			}
		}
		return phi;
	}
	
	/**
	 * count the number of each counting variable.
	 */
	public void doTopicCounting(){
		for( int i = 0; i < D; ++ i ){
			for( int j = 0; j < docs[i].length; ++ j ){
				//w is the word id
				int w = docs[i][j];
				nzw[topics[i][j]][w] ++;
				nzd[topics[i][j]][i] ++;
			}
		}
		
		for( int k = 0; k < K; ++ k ){
			cnz[k] = 0;
			/*cumulative sum of nzw[k][w] on w*/
			for( int w = 0; w < V; ++ w ){
				cnz[k] += nzw[k][w];
			}
//			System.out.println("cummulative nzw["+k+"][*] = " + cnz[k]);
		}
		
		for( int m = 0; m < D; ++ m ){
			/*cumulative sum of nzd[k][m] on k*/
			cnd[m] = docs[m].length;
//			System.out.println("cummulative nzd[*]["+m+"] = " + cnd[m]);
		}
	}
	
	/**
	 * random assign topic to each word using multinomial
	 */
	public void initSampleTopics(){
		
		topics = new int[D][];
		for( int i = 0; i < D; ++ i ){
			topics[i] = new int[docs[i].length];
			for( int j = 0; j < docs[i].length; ++ j ){
				/** initially multinomial distribution for all topic*/
				topics[i][j] = (int)(rand.nextDouble()*K);
//				topics[i][j] = (int)(Math.random()*K);
			}
		}
		
	}
	public static void main(String args[]){
		// words in documents  
        int[][] documents = { {1, 4, 3, 2, 3, 1, 4, 3, 2, 3, 1, 4, 3, 2, 3, 6},  
            {2, 2, 4, 2, 4, 2, 2, 2, 2, 4, 2, 2},  
            {1, 6, 5, 6, 0, 1, 6, 5, 6, 0, 1, 6, 5, 6, 0, 0},  
            {5, 6, 6, 2, 3, 3, 6, 5, 6, 2, 2, 6, 5, 6, 6, 6, 0},  
            {2, 2, 4, 4, 4, 4, 1, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 0},  
            {5, 4, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2}};  
        int M = documents.length;
        int K = 2;
        int V = 7;
        LDAGibbsTraining lda = new LDAGibbsTraining(documents,V);
        lda.config(K,2000,10000,10);
        
        lda.gibbsSample(2, .5);
        //输出模型参数，论文中式 （81）与（82）  
        double[][] theta = lda.normalizeTheta();
        double[][] phi = lda.normalizePhi();  
        System.out.println("theta: ");
        for( int i = 0; i < M; ++ i ){
        	for( int k = 0; k < K; ++ k ){
        		System.out.print(theta[k][i] + " ");
        	}
        	System.out.println();
        }
        
        
        System.out.println("phi: " );
        for( int k = 0; k < K; ++ k ){
        	for( int j = 0 ;  j < V; ++ j ){
        		System.out.print(phi[j][k] + " ");
        	}
        	System.out.println();
        }
	}
}
