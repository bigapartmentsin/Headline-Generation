package com.sohu.headgen;

import com.sohu.headgen.model.HeadlineGenerator;
import com.sohu.headgen.util.Summarizer;
import opennlp.tools.tokenize.Tokenizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.sohu.headgen.model.components.impl.AdvancedGraphWeigher;
import com.sohu.headgen.model.components.impl.DefaultGraphEncoder;
import com.sohu.headgen.model.components.impl.DefaultPathCompressor;
import com.sohu.headgen.util.OpenNLP;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * TODO Replace with proper description...
 * <p>
 * Created by stefano on 23/01/2017.
 */
public class Main {

    private static final Logger logger = LoggerFactory.getLogger(Main.class);

    public static List<String> readTxt(File file) {
        List<String> result = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(file));
            String s = null;
            while ((s = br.readLine()) != null) {
                JSONObject jsonObject = new JSONObject(s);
                String article = (String)jsonObject.get("content");
                article = article.replaceAll("(\r\n|\r|\n|\n\r)", " ");
                article = article.replaceAll("…", " ");
                article = article.replaceAll("\\.", ". ");
                result.add(article);
            }
            br.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return result;

    }

    public static void writeTxt(File file, String s) throws IOException {
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(file));
            bw.write(s);
            bw.newLine();
            bw.flush();
            bw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws IOException {
        Path folder = Paths.get(args[0]);

        List<String> sentences = Arrays.asList(
                "The wife of a former U.S. president Bill Clinton, Hillary Clinton, visited China last Monday.",
                "Hillary Clinton wanted to visit China last month but postponed her plans till Monday last week.",
                "Hillary Clinton paid a visit to the People Republic of China on Monday.",
                "Last week the Secretary State Ms. Clinton visited Chinese officials.");

        Collection<String> stopWords = Arrays.asList("a", "able", "about", "above", "after", "all", "also", "an",
                "and", "any", "as", "ask", "at", "back", "bad", "be", "because", "beneath", "big", "but", "by",
                "call", "can", "case", "child", "come", "company", "could", "day", "different", "do", "early", "even",
                "eye", "fact", "feel", "few", "find", "first", "for", "from", "get", "give", "go", "good",
                "government", "great", "group", "hand", "have", "he", "her", "high", "him", "his", "how", "i", "if",
                "important", "in", "into", "it", "its", "just", "know", "large", "last", "leave", "life", "like",
                "little", "long", "look", "make", "man", "me", "most", "my", "new", "next", "no", "not", "now",
                "number", "of", "old", "on", "one", "only", "or", "other", "our", "out", "over", "own", "part",
                "people", "person", "place", "point", "problem", "public", "right", "same", "say", "see", "seem",
                "she", "small", "so", "some", "take", "tell", "than", "that", "the", "their", "them", "then", "there",
                "these", "they", "thing", "think", "this", "time", "to", "try", "two", "under", "up", "us", "use",
                "want", "way", "we", "week", "well", "what", "when", "which", "who", "will", "with", "woman", "work",
                "world", "would", "year", "you", "young", "your");

        Summarizer summarizer = new Summarizer();

        List<String> summary = summarizer.getSummary("256679269", 4, 0.1, 0.01);

        HeadlineGenerator generator = HeadlineGenerator.builder()
                .on(folder)
                .withEncoder(new DefaultGraphEncoder())
                .withWeigher(new AdvancedGraphWeigher())
                .withCompressor(new DefaultPathCompressor())
                .build();

        Optional<String> headline = generator.process(summary, stopWords);
        String result;
        if (headline.isPresent()) {
            result = headline.get();
            System.out.println(" >> " + result);
        }
    }

}
