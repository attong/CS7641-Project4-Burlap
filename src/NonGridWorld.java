import burlap.behavior.functionapproximation.dense.DenseCrossProductFeatures;
import burlap.behavior.functionapproximation.dense.NormalizedVariableFeatures;
import burlap.behavior.functionapproximation.dense.fourier.FourierBasis;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
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

	private HashableStateFactory hashingFactory;
	
	public NonGridWorld(int levelnum) {
		this.bd = new BlockDude();
		this.bd.setRf(new RF());
		this.bd.setTf(new BlockDudeTF());
		this.domain = this.bd.generateDomain();
		if (levelnum==1) {
			this.st = (BlockDudeState) BlockDudeLevelConstructor.getLevel1(domain);	
		} else if  (levelnum==2){
			this.st = (BlockDudeState) BlockDudeLevelConstructor.getLevel2(domain);	
		} else {
			this.st = (BlockDudeState) BlockDudeLevelConstructor.getLevel3(domain);	
		}
		this.hashingFactory = new SimpleHashableStateFactory();
	}

	public void valueIterationExample(String outputPath){

		Planner planner = new ValueIteration(domain, 0.99, hashingFactory, 0.001, 100, "nongridvi");
		Policy p = planner.planFromState(st);
		planner.toggleDebugPrinting(true);

		PolicyUtils.rollout(p, st, domain.getModel()).write(outputPath + "vi");


	}
	
	public static void main(String[] args) {
		NonGridWorld world = new NonGridWorld(3);
		world.valueIterationExample("csv/nongridvi");
//
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
