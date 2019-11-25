import burlap.behavior.policy.GreedyQPolicy;
import java.util.ArrayList;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.ArrowActionGlyph;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.LandmarkColorBlendInterpolation;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.PolicyGlyphPainter2D;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.StateValuePainter2D;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.learning.tdmethods.SarsaLam;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.deterministic.DeterministicPlanner;
import burlap.behavior.singleagent.planning.deterministic.informed.Heuristic;
import burlap.behavior.singleagent.planning.deterministic.informed.astar.AStar;
import burlap.behavior.singleagent.planning.deterministic.uninformed.bfs.BFS;
import burlap.behavior.singleagent.planning.deterministic.uninformed.dfs.DFS;
import burlap.behavior.valuefunction.QProvider;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldRewardFunction;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.state.State;
import burlap.mdp.core.state.vardomain.VariableDomain;
import burlap.mdp.singleagent.common.GoalBasedRF;
import burlap.mdp.singleagent.common.VisualActionObserver;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;

import java.awt.*;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Scanner;

import com.opencsv.CSVWriter;
public class GridWorld {
	
	
	GridWorldDomain gwdg;
	OOSADomain domain;
	RewardFunction rf;
	TerminalFunction tf;
	StateConditionTest goalCondition;
	State initialState;
	HashableStateFactory hashingFactory;
	SimulatedEnvironment env;
	
	public GridWorldDomain createFourRooms(int size) {
		GridWorldDomain grid = new GridWorldDomain(size,size);
		grid.horizontalWall(0, 0, size/2);
		grid.horizontalWall(2, size/2, size/2);
		grid.horizontalWall(size/2+1,size-3, size/2-1);
		grid.horizontalWall(size-1,size-1, size/2-1);
		grid.verticalWall(0, 0, size/2);
		grid.verticalWall(2, size-3, size/2);
		grid.verticalWall(size-1, size-1, size/2);

		return grid;
	}
	
	public GridWorld() {
		int size = 10;

//		gwdg = new GridWorldDomain(2, 11);
		gwdg= createFourRooms(size);
//		gwdg.setMapToFourRooms();
		tf = new GridWorldTerminalFunction(size-1, size-1);
		gwdg.setTf(tf);
		GridWorldRewardFunction rf = new GridWorldRewardFunction(size,size,-1);
		rf.setReward(size-1, size-1, 0);
		gwdg.setRf(rf);
		goalCondition = new TFGoalCondition(tf);
		domain = gwdg.generateDomain();

		initialState = new GridWorldState(new GridAgent(0, 0), new GridLocation(size-1, size-1, "loc0"));
		hashingFactory = new SimpleHashableStateFactory();

		env = new SimulatedEnvironment(domain, initialState);
	}
	public GridWorld(int size) {

//		gwdg = new GridWorldDomain(2, 11);
		gwdg= createFourRooms(size);
//		gwdg.setMapToFourRooms();
		tf = new GridWorldTerminalFunction(size-1, size-1);
		gwdg.setTf(tf);
		goalCondition = new TFGoalCondition(tf);
		domain = gwdg.generateDomain();

		initialState = new GridWorldState(new GridAgent(0, 0), new GridLocation(size-1, size-1, "loc0"));
		hashingFactory = new SimpleHashableStateFactory();

		env = new SimulatedEnvironment(domain, initialState);
	}

	public void visualize(String outputpath){
		Visualizer v = GridWorldVisualizer.getVisualizer(gwdg.getMap());
		new EpisodeSequenceVisualizer(v, domain, outputpath);
	}


	public Planner valueIterationExample(String outputPath){

		Planner planner = new ValueIteration(domain, 0.99, hashingFactory, 0.001, 100, outputPath);
		planner.toggleDebugPrinting(true);
		Policy p = planner.planFromState(initialState);

		Episode episode = PolicyUtils.rollout(p, initialState, domain.getModel());
//		episode.write(outputPath + "vi");
		helper.printEpisodeStats(episode);
//		simpleValueFunctionVis((ValueFunction)planner, p);
		//manualValueFunctionVis((ValueFunction)planner, p);
		return planner;

	}


	public void qLearningExample(String outputPath){

		/** q learning params
		 * QLearning(SADomain domain, double gamma, HashableStateFactory hashingFactory, double qInit, double learningRate)
		 * @param domain the domain in which to learn
		 * @param gamma the discount factor
		 * @param hashingFactory the state hashing factory to use for Q-lookups
		 * @param qInit a {@link burlap.behavior.valuefunction.QFunction} object that can be used to initialize the Q-values.
		 * @param learningRate the learning rate
		 * @param learningPolicy the learning policy to follow during a learning episode.
		 * @param maxEpisodeSize the maximum number of steps the agent will take in a learning episode for the agent stops trying.
		 */
		File file = new File("csv/gridworldqlearning.csv"); 
        Scanner sc = new Scanner(System.in); 
        try { 
            // create FileWriter object with file as parameter 
            FileWriter outputfile = new FileWriter(file); 
  
            // create CSVWriter with ';' as separator 
            CSVWriter writer = new CSVWriter(outputfile, ',', 
                                             CSVWriter.NO_QUOTE_CHARACTER, 
                                             CSVWriter.DEFAULT_ESCAPE_CHARACTER, 
                                             CSVWriter.DEFAULT_LINE_END); 
            LearningAgent agent = new QLearning(domain, 0.99, hashingFactory, 0., 0.99);
			writer.writeNext(new String[] {"Episode", "steps","rewards"});
    		
    		//run learning for 50 episodes
    		for(int i = 0; i < 10000; i++){
    			Episode e = agent.runLearningEpisode(env, 1000);

//    			e.write(outputPath + "ql_" + i);
    			System.out.println(i + ": " + e.numTimeSteps());
//    			System.out.println(i+": " +((QLearning)agent).getMaxQChangeInLastEpisode());
//    			helper.printEpisodeStats(e);
//    			System.out.println(helper.gettotalreward(e));
//    			System.out.println(e.numTimeSteps());
    			
    			//reset environment for next learning episode
    			writer.writeNext(new String[] {String.valueOf(i+1), String.valueOf(e.numTimeSteps()), String.valueOf(helper.gettotalreward(e))});
    			env.resetEnvironment();
    		}
    		writer.close();
        }
        catch (IOException e) { 
            // TODO Auto-generated catch block 
            e.printStackTrace();
        }
		

	}
	
	public void policyIterationExample(String filnam) {
		Planner planner = new PolicyIteration(domain, 0.99, hashingFactory,0.001,10000,100,"gridpi");
		planner.toggleDebugPrinting(true);
		Policy p = planner.planFromState(initialState);

		Episode episode = PolicyUtils.rollout(p, initialState, domain.getModel());
//		episode.write(outputPath + "vi");
		helper.printEpisodeStats(episode);
		
	}



	public void simpleValueFunctionVis(ValueFunction valueFunction, Policy p){

		List<State> allStates = StateReachability.getReachableStates(
			initialState, domain, hashingFactory);
		ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(
			allStates, 11, 11, valueFunction, p);
		gui.initGUI();

	}

	public void manualValueFunctionVis(ValueFunction valueFunction, Policy p){

		List<State> allStates = StateReachability.getReachableStates(
			initialState, domain, hashingFactory);

		//define color function
		LandmarkColorBlendInterpolation rb = new LandmarkColorBlendInterpolation();
		rb.addNextLandMark(0., Color.RED);
		rb.addNextLandMark(1., Color.BLUE);

		//define a 2D painter of state values, 
		//specifying which attributes correspond to the x and y coordinates of the canvas
		StateValuePainter2D svp = new StateValuePainter2D(rb);
		svp.setXYKeys("agent:x", "agent:y", 
			new VariableDomain(0, 11), new VariableDomain(0, 11), 
			1, 1);

		//create our ValueFunctionVisualizer that paints for all states
		//using the ValueFunction source and the state value painter we defined
		ValueFunctionVisualizerGUI gui = new ValueFunctionVisualizerGUI(
			allStates, svp, valueFunction);

		//define a policy painter that uses arrow glyphs for each of the grid world actions
		PolicyGlyphPainter2D spp = new PolicyGlyphPainter2D();
		spp.setXYKeys("agent:x", "agent:y", new VariableDomain(0, 11), 
			new VariableDomain(0, 11), 
			1, 1);

		spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_NORTH, new ArrowActionGlyph(0));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_SOUTH, new ArrowActionGlyph(1));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_EAST, new ArrowActionGlyph(2));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTION_WEST, new ArrowActionGlyph(3));
		spp.setRenderStyle(PolicyGlyphPainter2D.PolicyGlyphRenderStyle.DISTSCALED);


		//add our policy renderer to it
		gui.setSpp(spp);
		gui.setPolicy(p);

		//set the background color for places where states are not rendered to grey
		gui.setBgColor(Color.GRAY);

		//start it
		gui.initGUI();



	}
	
	public void varySize() {
		for (int i = 5;i <100;i+=5) {
			GridWorld gw = new GridWorld(i);
			gw.valueIterationExample("vi_l"+String.valueOf(i));
		}
		
	}

	public void varyGamma(boolean pi) {
		double[] gammas = {0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99};
		File file = new File("csv/gridworldvigammatest.csv"); 
		if (pi==true) {
			
		}
        Scanner sc = new Scanner(System.in); 
        try { 
            // create FileWriter object with file as parameter 
            FileWriter outputfile = new FileWriter(file); 
  
            // create CSVWriter with ';' as separator 
            CSVWriter writer = new CSVWriter(outputfile, ',', 
                                             CSVWriter.NO_QUOTE_CHARACTER, 
                                             CSVWriter.DEFAULT_ESCAPE_CHARACTER, 
                                             CSVWriter.DEFAULT_LINE_END); 

            writer.writeNext(new String[]{"Gamma","Stepstogoal","numIterations","wallclockruntime"});
    		for (int i = 0; i<gammas.length;i++) {
    			for (int j = 0; j<3;j++) {
	    			Planner planner = new ValueIteration(domain, gammas[i], hashingFactory, 0.001, 100, "gridvi");
	    			planner.toggleDebugPrinting(true);
	    			Policy p = planner.planFromState(initialState);
	    			Episode episode = PolicyUtils.rollout(p, initialState, domain.getModel());
	    			int steps = helper.printEpisodeStats(episode);
	    			int iterations=((ValueIteration)planner).getConvergenceIterations();
	    			double wallclock= ((ValueIteration)planner).getRunTime();
	    			writer.writeNext(new String[] {String.valueOf(gammas[i]),String.valueOf(steps),
	    					String.valueOf(iterations),String.valueOf(wallclock)});
    			}
    		}
            writer.close(); 
        } 
        catch (IOException e) { 
            // TODO Auto-generated catch block 
            e.printStackTrace(); 
        } 
	}
	public void experimentAndPlotter(){

		//different reward function for more structured performance plots
		((FactoredModel)domain.getModel()).setRf(new GoalBasedRF(this.goalCondition, 5.0, -0.1));

		/**
		 * Create factories for Q-learning agent and SARSA agent to compare
		 */
		LearningAgentFactory qLearningFactory = new LearningAgentFactory() {

			public String getAgentName() {
				return "Q-Learning";
			}


			public LearningAgent generateAgent() {
				return new QLearning(domain, 0.99, hashingFactory, 0.3, 0.1);
//				return new ValueIteration(domain, 0.99, hashingFactory, 0.001, 100);
			}
		};

		LearningAgentFactory sarsaLearningFactory = new LearningAgentFactory() {

			public String getAgentName() {
				return "SARSA";
			}


			public LearningAgent generateAgent() {
				return new SarsaLam(domain, 0.99, hashingFactory, 0.0, 0.1, 1.);
			}
		};

		LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(
			env, 10, 100, qLearningFactory, sarsaLearningFactory);
		exp.setUpPlottingConfiguration(500, 250, 2, 1000,
				TrialMode.MOST_RECENT_AND_AVERAGE,
				PerformanceMetric.CUMULATIVE_STEPS_PER_EPISODE,
				PerformanceMetric.AVERAGE_EPISODE_REWARD);

		exp.startExperiment();
		exp.writeStepAndEpisodeDataToCSV("expData");

	}


	public static void main(String[] args) {
		GridWorld grid = new GridWorld();
		final String outputPath = "csv/";
//		grid.valueIterationExample("gridvi");
//		grid.varyGamma();
//		grid.policyIterationExample("gridpi");
		grid.qLearningExample("csv/");
		
//		grid.experimentAndPlotter();
//		grid.visualize(outputPath);

	}
	
}
