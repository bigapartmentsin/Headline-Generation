package com.sohu.headgen.util;

import com.hankcs.hanlp.HanLP;
import com.sohu.headgen.model.SummarizationContentEntity;
import net.dongliu.requests.Requests;
import opennlp.tools.postag.POSTagger;
import opennlp.tools.sentdetect.SentenceDetector;
import opennlp.tools.tokenize.Tokenizer;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.*;
import java.util.regex.*;


//提供摘要服务接口，本服务基于LexRank算法
public class Summarizer{

    private static final Tokenizer TOKENIZER = OpenNLP.getTokenizer();
    private static final POSTagger TAGGER = OpenNLP.getPOSTagger();
    private static final SentenceDetector DETECTOR = OpenNLP.getSentenceDetector();

    private double avgsl;
    private static final Double PRECISION = 0.000000001;
    private static final Double k1 = 1.2;
    private static final Double b = 0.75;


    //获取摘要
    public List<String> getSummary(String mpId, int count, double threshold, double epsilon) {


        //将分词的内容组成句子
        SummarizationContentEntity summarizationContentEntity = mergeSentenceWords(mpId);

        //计算每个句子中各词的TF
        List<HashMap<String, Double>> tfMetrics = computeTfMetrics(summarizationContentEntity);
        //计算所有词的IDF
        HashMap<String, Double> idfMetrics = computeIdfMetrics(summarizationContentEntity, tfMetrics);
        //计算所有句子相互之间相似度的邻接矩阵
        double[][] adjacencyMatrix = createAdjacencyMatrix(summarizationContentEntity, threshold, tfMetrics, idfMetrics);
        //计算所有句子的分数
        double[] scores = powerMethod(adjacencyMatrix, epsilon);
        //按分数和顺序提取固定字数的摘要
        return getBestSenteces(summarizationContentEntity, scores, count);

    }

    private List<String> getBestSenteces(SummarizationContentEntity summarizationContentEntity, double[] scores, int count) {
        List<String> result = new ArrayList<>();
        List<String> sentences = summarizationContentEntity.getSentences();
        HashMap<Integer, Double> infos = new HashMap<>();
        int sentencesCount = summarizationContentEntity.getSentencesCount();
        //将句子的order和score组合成map
        for (int i = 0; i < sentencesCount; i++) {
            infos.put(i, scores[i]);
        }
        //将map转换成list并按分数排序
        List<Map.Entry<Integer, Double>> infoEntries = new ArrayList<>(infos.entrySet());
        Collections.sort(infoEntries, (o1, o2) -> o2.getValue().compareTo(o1.getValue()));
        int curCount = 0;
        int sentenceIndex;
        //按照字数限制选取分数较高的句子
        for (int i = 0; i < sentencesCount; i++) {
            if (curCount >= count) {
                for (int j = infoEntries.size() - 1; j >= i; j--) {
                    infoEntries.remove(j);
                }
                break;
            }
            sentenceIndex = infoEntries.get(i).getKey();
            String sentence = sentences.get(sentenceIndex);
            sentences.set(sentenceIndex, sentence);
            curCount++;
        }
        //再次按照order排序句子
        Collections.sort(infoEntries, (o1, o2) -> o1.getKey().compareTo(o2.getKey()));
        //转换成字符串并截取规定字数
        infoEntries.forEach(info -> result.add(sentences.get(info.getKey())));

        return result;
    }

    private List<HashMap<String, Double>> computeTfMetrics(SummarizationContentEntity summarizationContentEntity) {
        List<HashMap<String, Double>> tfMetrics = new ArrayList<>();
        int sentencesCount = summarizationContentEntity.getSentencesCount();
        for (int i = 0; i < sentencesCount; i++) {
            List<String> sentenceWords = summarizationContentEntity.getSentenceWords(i);
            HashMap<String, Double> wordsCountMap = new HashMap<>();
            double maxCount = 1.0;
            double count;
            //统计每个句子中各词的词频
            for (int j = 0; j < sentenceWords.size(); j++) {
                String curWord = sentenceWords.get(j);
                if (wordsCountMap.containsKey(sentenceWords.get(j))) {
                    count = wordsCountMap.get(curWord);
                    count += 1.0;
                } else {
                    count = 1.0;
                }
                wordsCountMap.put(curWord, count);
                if (count > maxCount) {
                    maxCount = count;
                }

            }
            Set<String> keys = wordsCountMap.keySet();
            for (String word : keys) {
                count = wordsCountMap.get(word);
                count /= maxCount;
                wordsCountMap.put(word, count);
            }
            tfMetrics.add(wordsCountMap);
        }

        return tfMetrics;
    }

    private HashMap<String, Double> computeIdfMetrics(SummarizationContentEntity summarizationContentEntity,
                                                      List<HashMap<String, Double>> tfMetrics) {
        HashMap<String, Double> idfMetrics = new HashMap<>();
        int sentencesCount = summarizationContentEntity.getSentencesCount();
        for (String word : summarizationContentEntity.getWordsSet()) {
            int count = 0;
            for (int i = 0; i < sentencesCount; i++) {
                if (tfMetrics.get(i).containsKey(word)) {
                    count++;
                }
            }
            double idf = Math.log((double) sentencesCount / (double) (1 + count));
            idfMetrics.put(word, idf);
        }
        return idfMetrics;
    }

    private double[][] createAdjacencyMatrix(SummarizationContentEntity summarizationContentEntity,
                                             double threshold, List<HashMap<String, Double>> tfMetrics,
                                             HashMap<String, Double> idfMetrics) {
        int sentencesCount = summarizationContentEntity.getSentencesCount();
        int l1, l2;
        double[][] matrix = new double[sentencesCount][sentencesCount];
        double[] degrees = new double[sentencesCount];

        for (int row = 0; row < sentencesCount; row++) {
            for (int col = 0; col < sentencesCount; col++) {
                //计算两个句子的相似度
//                matrix[row][col] = cosineSimilarity(tfMetrics.get(row), tfMetrics.get(col), idfMetrics);
                l1 = summarizationContentEntity.getSentenceWords(row).size();
                l2 = summarizationContentEntity.getSentenceWords(col).size();
                matrix[row][col] = bm25CosineSimilarity(tfMetrics.get(row), tfMetrics.get(col), idfMetrics, l1, l2);

                //如果大于阈值认为两个句子相关，否则不相关
                if (matrix[row][col] > threshold) {
                    matrix[row][col] = 1.0;
                    degrees[row] += 1.0;
                } else {
                    matrix[row][col] = 0.0;
                }
            }
        }
        for (int row = 0; row < sentencesCount; row++) {
            for (int col = 0; col < sentencesCount; col++) {
                if (Math.abs(degrees[row]) < PRECISION) { //degrees[row] == 0
                    degrees[row] = 1.0;
                }
                matrix[row][col] /= degrees[row];
            }
        }

        return matrix;

    }

    private double[] powerMethod(double[][] matrix, double epsilon) {
        RealMatrix adjacencyMatrix = new Array2DRowRealMatrix(matrix);
        int sentencesCount = adjacencyMatrix.getRowDimension();
        //初始化分数向量
        RealVector pVector = new ArrayRealVector(sentencesCount, 1.0 / (double) sentencesCount);
        double lambda = 1.0;
        //距离大于epsilon继续迭代
        while (lambda > epsilon) {
            RealVector pVectorNext = adjacencyMatrix.transpose().operate(pVector);
            lambda = pVector.getDistance(pVectorNext);
            pVector = pVectorNext;
        }

        return pVector.toArray();
    }

    //相似度计算
    private double cosineSimilarity(HashMap<String, Double> tf1, HashMap<String, Double> tf2,
                                    HashMap<String, Double> idfMetrics) {
        /*    S_i = TF_i*IDF_i
         *
         *                S_i*S_j
         *   Sim_ij = ---------------
         *              |S_i|*|S_j|
         */
        HashSet<String> commonWords = new HashSet<>();
        commonWords.addAll(tf1.keySet());
        commonWords.retainAll(tf2.keySet());

        double numerator = 0.0;
        double denominator1 = 0.0;
        double denominator2 = 0.0;

        for (String word : commonWords) {
            numerator += tf1.get(word) * tf2.get(word) * Math.pow(idfMetrics.get(word), 2);
        }

        for (String word : tf1.keySet()) {
            denominator1 += Math.pow(tf1.get(word) * idfMetrics.get(word), 2);
        }
        for (String word : tf2.keySet()) {
            denominator2 += Math.pow(tf2.get(word) * idfMetrics.get(word), 2);
        }

        if (denominator1 > 0 && denominator2 > 0) {
            return numerator / (Math.sqrt(denominator1) * Math.sqrt(denominator2));
        } else {
            return 0.0;
        }
    }

    private double bm25CosineSimilarity(HashMap<String, Double> tf1, HashMap<String, Double> tf2,
                                        HashMap<String, Double> idfMetrics, int l1, int l2) {
        HashSet<String> commonWords = new HashSet<>();
        commonWords.addAll(tf1.keySet());
        commonWords.retainAll(tf2.keySet());

        double numerator = 0.0;
        double denominator1 = 0.0;
        double denominator2 = 0.0;

        for (String word:commonWords) {
            double factor1 = tf1.get(word) * (k1 + 1.0) / (tf1.get(word) + k1 * (1 - b + b * (double)l1 / avgsl));
            double factor2 = tf2.get(word) * (k1 + 1.0) / (tf2.get(word) + k1 * (1 - b + b * (double)l2 / avgsl));
            numerator += factor1 * factor2 * Math.pow(idfMetrics.get(word), 2);
        }
        for (String word:tf1.keySet()) {
            denominator1 += Math.pow(tf1.get(word) * (k1 + 1.0) / (tf1.get(word) + k1 * (1 - b + b * (double)l1 / avgsl)), 2);
        }
        for (String word:tf2.keySet()) {
            denominator2 += Math.pow(tf2.get(word) * (k1 + 1.0) / (tf2.get(word) + k1 * (1 - b + b * (double)l2 / avgsl)), 2);
        }

        if (denominator1 > 0 && denominator2 > 0) {
            return numerator / (Math.sqrt(denominator1) * Math.sqrt(denominator2));
        }
        else {
            return 0.0;
        }
    }

    //按照句子规则组合句子
    private SummarizationContentEntity mergeSentenceWords(String mpId) {
        String url = "http://10.16.57.57:8887/rec/content/segmentation/v1?mpId=" + mpId;
        String response = Requests.get(url).send().readToText();
        JSONObject jsonObject = new JSONObject(response);
        JSONObject data = jsonObject.getJSONObject("data");
        JSONArray contentWords = data.getJSONArray("content");
        double totalSentencesLength = 0;
        StringBuilder sentenceBuilder = new StringBuilder();
        List<String> sentenceWords = new ArrayList<>();
        boolean flag = false;
        SummarizationContentEntity summarizationContentEntity = new SummarizationContentEntity();
        for (int i = 0; i < contentWords.length(); i++) {
            JSONObject contentWord = (JSONObject)contentWords.get(i);
            String curWord = contentWord.getString("word");
            String curWordType = contentWord.getString("posType");
            if (curWord.equals("　　")) {
                continue;
            }
            if (!curWord.equals("\n")) {
                sentenceBuilder.append(curWord);
            }
            if (!curWordType.equals("w")
                    && !curWordType.equals("mq")
                    && !curWordType.equals("x")) {
                sentenceWords.add(curWord);
                summarizationContentEntity.addWord(curWord);
            }
            if (curWord.equals("。") || curWord.equals("！")
                    || curWord.equals("？") || curWord.charAt(curWord.length() - 1) == '\n') {
                if (sentenceWords.size() > 0) {
                    summarizationContentEntity.addSentence(sentenceBuilder.toString(), sentenceWords);
                    totalSentencesLength += sentenceWords.size();
                    flag = true;
                }

            }
            if (flag) {
                sentenceBuilder = new StringBuilder();
                sentenceWords = new ArrayList<>();
                flag = false;
            }
        }
        if (sentenceWords.size() > 0) {
            summarizationContentEntity.addSentence(sentenceBuilder.toString(), sentenceWords);
            totalSentencesLength += sentenceWords.size();
        }
        this.avgsl = totalSentencesLength / (double)summarizationContentEntity.getSentencesCount();
        return summarizationContentEntity;
    }

}
