import java.util.List;

import burlap.behavior.singleagent.Episode;

//helper functions class
public class helper {
	
	public static int printEpisodeStats(Episode episode) {
		System.out.println(episode.actionString());
		List<Double> temp = episode.rewardSequence;
		System.out.println("Number of steps: " + String.valueOf(temp.size()));
		System.out.println("Final State Reward: "+ temp.get(temp.size()-1).toString());
		return(temp.size());
	}
	
	public static double gettotalreward(Episode episode) {
		List<Double> rewards= episode.rewardSequence;
		double sum = 0;
		for (double i :rewards) {
			sum+=i;
		}
		return sum;
	}
	
	public static int getNumSteps(Episode episode) {
		return episode.numTimeSteps();
	}

}
