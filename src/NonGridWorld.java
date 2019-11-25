import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

import com.opencsv.CSVWriter;

import burlap.mdp.singleagent.model.FactoredModel;
import burlap.behavior.functionapproximation.dense.DenseCrossProductFeatures;
import burlap.behavior.functionapproximation.dense.NormalizedVariableFeatures;
import burlap.behavior.functionapproximation.dense.fourier.FourierBasis;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.lspi.LSPI;
import burlap.behavior.singleagent.learning.lspi.SARSCollector;
import burlap.behavior.singleagent.learning.lspi.SARSData;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.deterministic.informed.NullHeuristic;
import burlap.behavior.singleagent.planning.deterministic.informed.astar.AStar;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.blockdude.BlockDude;
import burlap.domain.singleagent.blockdude.BlockDudeLevelConstructor;
import burlap.domain.singleagent.blockdude.BlockDudeTF;
import burlap.domain.singleagent.blockdude.BlockDudeVisualizer;
import burlap.domain.singleagent.blockdude.state.BlockDudeCell;
import burlap.domain.singleagent.blockdude.state.BlockDudeState;
import burlap.domain.singleagent.blocksworld.BlocksWorld;
import burlap.domain.singleagent.blocksworld.BlocksWorldState;
import burlap.domain.singleagent.blocksworld.BlocksWorldVisualizer;
import burlap.domain.singleagent.cartpole.CartPoleDomain;
import burlap.domain.singleagent.mountaincar.MCRandomStateGenerator;
import burlap.domain.singleagent.mountaincar.MountainCar;
import burlap.domain.singleagent.pomdp.tiger.TigerDomain;
import burlap.mdp.auxiliary.StateGenerator;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.Domain;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;
import burlap.mdp.core.state.vardomain.VariableDomain;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.common.GoalBasedRF;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.mdp.singleagent.pomdp.PODomain;
import burlap.shell.EnvironmentShell;
import burlap.shell.visual.VisualExplorer;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;

public class NonGridWorld {

	/**
	 * Name for the up action
	 */
	public static final String ACTION_UP = "up";

	/**
	 * Name for the east action
	 */
	public static final String ACTION_EAST = "east";

	/**
	 * Name for the west action
	 */
	public static final String ACTION_WEST = "west";

	/**
	 * Name for the pickup action
	 */
	public static final String ACTION_PICKUP = "pickup";

	/**
	 * Name for the put down action
	 */
	public static final String ACTION_PUT_DOWN = "putdown";

	public static BlockDudeState st;
	
	public BlockDude bd;
	private SADomain domain;
	StateConditionTest goalCondition;

	private HashableStateFactory hashingFactory;

	SimulatedEnvironment env;
	
	public NonGridWorld(int levelnum) {
		this.bd = new BlockDude();
		this.bd.setRf(new RF());
		this.bd.setTf(new BlockDudeTF());
		goalCondition = new TFGoalCondition(new BlockDudeTF());
		this.domain = this.bd.generateDomain();
		if (levelnum==1) {
			this.st = (BlockDudeState) BlockDudeLevelConstructor.getLevel1(domain);	
		} else if  (levelnum==2){
			this.st = (BlockDudeState) BlockDudeLevelConstructor.getLevel2(domain);	
		} else {
			this.st = (BlockDudeState) BlockDudeLevelConstructor.getLevel3(domain);	
		}
		this.hashingFactory = new SimpleHashableStateFactory();
		this.env = new SimulatedEnvironment(domain, st);
	}

	public void valueIterationExample(String outputPath){

		Planner planner = new ValueIteration(domain, 0.99, hashingFactory, 0.001, 100, "nongridvi");
		Policy p = planner.planFromState(st);
		planner.toggleDebugPrinting(true);

		PolicyUtils.rollout(p, st, domain.getModel()).write(outputPath + "vi");


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
				return new QLearning(domain, 0.99, hashingFactory, 0.3, 0.9);
			}
		};


		LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(
			env, 10, 100, qLearningFactory);
		exp.setUpPlottingConfiguration(500, 250, 2, 1000,
				TrialMode.MOST_RECENT_AND_AVERAGE,
				PerformanceMetric.CUMULATIVE_STEPS_PER_EPISODE,
				PerformanceMetric.AVERAGE_EPISODE_REWARD);

		exp.startExperiment();
		exp.writeStepAndEpisodeDataToCSV("expData");

	}
	
	public void policyIterationExample(String filnam){

		Planner planner = new PolicyIteration(domain, 0.99, hashingFactory, 0.001, 10000 , 100, filnam);
		Policy p = planner.planFromState(st);
		planner.toggleDebugPrinting(true);

//		PolicyUtils.rollout(p, st, domain.getModel()).write( + "vi");


	}
	
	public void qLearningExample(String outputPath){

		/** 
		 * @param domain the domain in which to learn
		 * @param gamma the discount factor
		 * @param hashingFactory the state hashing factory to use for Q-lookups
		 * @param qInit a {@link burlap.behavior.valuefunction.QFunction} object that can be used to initialize the Q-values.
		 * @param learningRate the learning rate
		 * @param learningPolicy the learning policy to follow during a learning episode.
		 * @param maxEpisodeSize the maximum number of steps the agent will take in a learning episode for the agent stops trying.
		 */
		LearningAgent agent = new QLearning(domain, 0.99, hashingFactory, 0., 0.99);

		//run learning for 50 episodes
		for(int i = 0; i < 10000; i++){
			Episode e = agent.runLearningEpisode(env, 1000);

//			e.write(outputPath + "ql_" + i);
			System.out.println(i + ": " + e.maxTimeStep());
			System.out.println(i+": " +((QLearning)agent).getMaxQChangeInLastEpisode());
			helper.printEpisodeStats(e);

			//reset environment for next learning episode
			env.resetEnvironment();
		}

	}
	
	public void varyGamma() {
		double[] gammas = {0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99};
		File file = new File("csv/nongridworldvigamma.csv"); 
        Scanner sc = new Scanner(System.in); 
        try { 
            // create FileWriter object with file as parameter 
            FileWriter outputfile = new FileWriter(file); 
  
            // create CSVWriter with ';' as separator 
            CSVWriter writer = new CSVWriter(outputfile, ',', 
                                             CSVWriter.NO_QUOTE_CHARACTER, 
                                             CSVWriter.DEFAULT_ESCAPE_CHARACTER, 
                                             CSVWriter.DEFAULT_LINE_END); 

            writer.writeNext(new String[]{"Gamma","Stepstogoal","numIterations"});
    		for (int i = 0; i<gammas.length;i++) {
    			for (int j = 0; j<3;j++) {
	    			Planner planner = new ValueIteration(domain, gammas[i], hashingFactory, 0.001, 100, "nongridvi");
	    			planner.toggleDebugPrinting(true);
	    			Policy p = planner.planFromState(st);
	    			Episode episode = PolicyUtils.rollout(p, st, domain.getModel());
	    			int steps = helper.printEpisodeStats(episode);
	    			int iterations=((ValueIteration)planner).getConvergenceIterations();
	    			writer.writeNext(new String[] {String.valueOf(gammas[i]),String.valueOf(steps),String.valueOf(iterations)});
    			}
    		}
            writer.close(); 
        } 
        catch (IOException e) { 
            // TODO Auto-generated catch block 
            e.printStackTrace(); 
        } 
	}
	public static void main(String[] args) {
		NonGridWorld world = new NonGridWorld(1);
//		world.valueIterationExample("nongridvi");
//		world.policyIterationExample("nongridpi");
//		world.varyGamma();
		
//		world.qLearningExample("csv/");
		world.experimentAndPlotter();
//		PolicyUtils.rollout(p, st, domain.getModel()).write( "csv/testvi");
//		Visualizer v = BlockDudeVisualizer.getVisualizer(bd.getMaxx(), bd.getMaxy());
//
//
//
//		VisualExplorer exp = new VisualExplorer(domain, v, st);
//
//		exp.addKeyAction("w", ACTION_UP, "");
//		exp.addKeyAction("d", ACTION_EAST, "");
//		exp.addKeyAction("a", ACTION_WEST, "");
//		exp.addKeyAction("s", ACTION_PICKUP, "");
//		exp.addKeyAction("x", ACTION_PUT_DOWN, "");

//		exp.initGUI();


	}
	public static class RF implements RewardFunction{

		public RF(){
		}

		@Override
		public double reward(State s, Action a, State sprime) {


			//are they at goal location?
			
			BlockDudeState bs = (BlockDudeState)sprime;

			int ax = bs.agent.x;
			int ay = bs.agent.y;

			int ex = bs.exit.x;
			int ey = bs.exit.y;

			if( ex == ax && ey == ay) {
				return 1000.;
			}else {
				return 0;
			}
		}
	}


}
