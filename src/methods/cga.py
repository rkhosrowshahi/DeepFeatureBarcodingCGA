import time
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.core.evaluator import Evaluator
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem
from .dhash import dhash_barcoding
from .dft import dft_barcoding
from ..utils import evaluate_retrieval


# Define the optimization problem
class FeatureOrderOptimization(Problem):
    def __init__(self, train_features, train_labels, val_features, val_labels, binarizer=None, seed=42):
        self.train_features = train_features
        self.train_labels = train_labels
        self.val_features = val_features
        self.val_labels = val_labels
        self.binarizer = binarizer
        if binarizer is None:
            self.binarizer = dhash_barcoding
        
        n_features = train_features.shape[1]
        batch_size = 128
        if batch_size > train_features.shape[0]:
            batch_size = train_features.shape[0]
        self.batch_size = batch_size
        self.seed = seed
        super().__init__(n_var=n_features, n_obj=1, xl=0, xu=n_features-1, vtype=int)

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((len(X),))
        for i, permutation in enumerate(X):
            train_batch, train_labels_batch = self.get_random_batch(self.train_features, self.train_labels, batch_size=self.batch_size)
            # Maximize accuracy (minimize negative accuracy)
            F[i] = self.mini_batch_evaluate(permutation, train_batch, train_labels_batch, self.val_features, self.val_labels)
        out["F"] = F

    def evaluate_retrieval_with_hit_or_miss(
        self,
        query_codes: np.ndarray,
        database_codes: np.ndarray,
        query_labels: np.ndarray,
        database_labels: np.ndarray,
        k: int = 5
    ) -> float:
        """
        Evaluate retrieval performance using Hamming distance and top-k accuracy.

        Args:
            query_codes (np.ndarray): Binary codes for query items, shape (n_queries, code_length)
            database_codes (np.ndarray): Binary codes for database items, shape (n_database, code_length)
            query_labels (np.ndarray): Labels for query items, shape (n_queries,)
            database_labels (np.ndarray): Labels for database items, shape (n_database,)
            k (int, optional): Number of nearest neighbors to consider. Defaults to 5.

        Returns:
            float: Mean retrieval accuracy across all queries
        """
        # Input validation
        if not all(isinstance(x, np.ndarray) for x in [query_codes, database_codes, query_labels, database_labels]):
            raise TypeError("All inputs must be numpy arrays")
        
        if len(query_codes) != len(query_labels):
            raise ValueError("Number of query codes and labels must match")
        if len(database_codes) != len(database_labels):
            raise ValueError("Number of database codes and labels must match")
        if query_codes.shape[1] != database_codes.shape[1]:
            raise ValueError("Code dimensions must match")
        if k < 1 or k > len(database_codes):
            raise ValueError(f"k must be between 1 and {len(database_codes)}")

        # Compute Hamming distances using bitwise XOR (faster than direct Hamming distance)
        query_codes = query_codes.astype(np.uint8)
        database_codes = database_codes.astype(np.uint8)

        # Compute Hamming distances efficiently using XOR
        distances = np.bitwise_xor(
            query_codes[:, np.newaxis, :], 
            database_codes[np.newaxis, :, :]
        ).sum(axis=2)

        # # Get top-k indices
        # top_k_indices = np.argpartition(distances, k, axis=1)[:, :k]
        # retrieved_labels = database_labels[top_k_indices]

        # # F1 Score (original majority voting)
        # n_queries = len(query_labels)
        # predicted_labels = np.zeros(n_queries, dtype=query_labels.dtype)
        # for i in range(n_queries):
        #     counts = np.bincount(retrieved_labels[i])
        #     predicted_labels[i] = counts.argmax()
        # f1 = f1_score(query_labels, predicted_labels, average="weighted")
        # return f1

        # Precision@k
        # precisions_at_k = np.mean(retrieved_labels == query_labels[:, np.newaxis], axis=1)
        # precision_k = np.mean(precisions_at_k)

        # mAP
        sorted_indices = np.argsort(distances, axis=1)
        retrieved_labels_full = database_labels[sorted_indices]
        relevant = (retrieved_labels_full == query_labels[:, np.newaxis]).astype(int)
        precisions = np.cumsum(relevant, axis=1) / (np.arange(relevant.shape[1]) + 1)
        ap = np.sum(precisions * relevant, axis=1) / np.maximum(np.sum(relevant, axis=1), 1)
        mean_ap = np.mean(ap)

        # return (mean_ap + precision_k + f1) / 3
        return mean_ap
        # return precision_k

    def mini_batch_evaluate(self, permutation, database_batch, database_labels_batch, query_batch, query_labels_batch, k=5, minimize=True):
        # print(f"Length of val_batch: {val_batch.shape[0]}")
        # Reorder features
        database_reordered_features = database_batch[:, permutation]
        query_reordered_features = query_batch[:, permutation]
        # Binarize with MinMax
        database_binary_codes = self.binarizer(database_reordered_features)
        query_binary_codes = self.binarizer(query_reordered_features)
        # Evaluate retrieval accuracy
        accuracy = self.evaluate_retrieval_with_hit_or_miss(query_codes=query_binary_codes, database_codes=database_binary_codes, 
                                      query_labels=query_labels_batch, database_labels=database_labels_batch, 
                                      k=k)
        if minimize:
            return -accuracy
        else:
            return accuracy
    
    def get_random_batch(self, inputs, targets, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if inputs.shape[0] > batch_size:
            batch_idx = np.random.choice(a=np.arange(inputs.shape[0]), size=batch_size, replace=False)
            # train_batch = inputs[batch_idx]
            # train_labels_batch = targets[batch_idx]
            return inputs[batch_idx], targets[batch_idx]
        else:
            return inputs, targets


class CombinatorialGeneticAlgorithm:
    def __init__(self, num_features, 
                 crossover_prob=0.9, mutation_prob=0.9, n_gen=50, pop_size=100,
                 train_features=None, train_labels=None, 
                 val_features=None, val_labels=None, 
                 test_features=None, test_labels=None, 
                 k=3, binarizer=None, seed=42):
        self.num_features = num_features
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.train_features = train_features
        self.train_labels = train_labels
        self.val_features = val_features
        self.val_labels = val_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.best_permutation = np.arange(num_features)
        self.k = k
        self.seed = seed
        if binarizer == "diff":
            self.binarizer = dhash_barcoding
        elif binarizer == "dft":
            self.binarizer = dft_barcoding
        else:
            raise ValueError(f"Binarizer {binarizer} not found")

        self.problem = FeatureOrderOptimization(train_features=self.train_features, train_labels=self.train_labels, 
                                                val_features=self.val_features, val_labels=self.val_labels, 
                                                binarizer=self.binarizer, seed=self.seed)

        self.best_fitnesses = []
        self.test_f1_scores = []
        self.test_precision_k = []
        self.test_mAP = []
        self.steps = []
        self.population = []
        self.best_solution = None
        self.population_fitness_scores = []
        self.recorded_steps = []
        self.best_fitness_score = 0

    def fit(self, train_features, train_labels, verbose=True): 
        # CGA setup

        algorithm = GA(
            pop_size=self.pop_size,
            sampling=PermutationRandomSampling(),
            crossover=OrderCrossover(prob=self.crossover_prob),
            mutation=InversionMutation(prob=self.mutation_prob),
            eliminate_duplicates=True if self.problem.n_var > 7 else False
        )
        termination = NoTermination()
        algorithm.setup(self.problem, termination=termination, seed=self.seed, verbose=verbose)

        # Optimization
        print("Running Combinatorial Genetic Algorithm...")
        start_time = time.time()
        # recorded_steps = []
        for t in range(self.n_gen):
            P = algorithm.ask()
            X = P.get("X")
            if t == 0:
                X[-1] = np.arange(self.num_features)
                P.set("X", X)
            # Get random batch of training data
            database_inputs, database_targets = self.problem.get_random_batch(self.train_features, self.train_labels, batch_size=128)
            # Get random batch of validation data
            query_inputs, query_targets = self.problem.get_random_batch(self.train_features, self.train_labels, batch_size=128)

            F = np.zeros((len(X),))
            for i, x in enumerate(X):
                F[i] = self.problem.mini_batch_evaluate(x, database_inputs, database_targets, query_inputs, query_targets, k=self.k)

            static = StaticProblem(self.problem, F=F)
            Evaluator().eval(static, P)
            algorithm.tell(infills=P)
            X = algorithm.pop.get("X")
            F = algorithm.pop.get("F")
            for i, x in enumerate(X):
                F[i] = self.problem.mini_batch_evaluate(x, database_inputs, database_targets, query_inputs, query_targets, k=self.k)
            algorithm.pop.set("F", F)
            

            res = algorithm.result()
            if t % 10 == 0 or t == self.n_gen - 1:
                best_permutation = res.X
                query_codes = self.transform(self.test_features, permutation=best_permutation)
                database_codes = self.transform(self.train_features, permutation=best_permutation)
                f1, precision_k, mean_ap = evaluate_retrieval(query_codes=query_codes, database_codes=database_codes, 
                                      query_labels=self.test_labels, database_labels=self.train_labels, k=self.k)
                print(f"Step {t+1} - F1: {f1:.4f}, Precision@k: {precision_k:.4f}, mAP: {mean_ap:.4f}")
                self.best_fitnesses.append(res.F.mean())
                self.recorded_steps.append(t)
                self.test_f1_scores.append(f1)
                self.test_precision_k.append(precision_k)
                self.test_mAP.append(mean_ap)


            if res.pop.get("F").mean() == -1.0:
                print("Found the best permutation!, stopping the optimization ...")
                break

        ga_time = time.time() - start_time
        print(f"Best permutation found in {ga_time:.2f}s")
        res = algorithm.result()
        best_permutation = res.X
        self.best_permutation = best_permutation
        self.population = res.pop.get("X")
        self.population_fitness_scores = res.pop.get("F")
        self.best_solution = res.X
        self.best_fitness_score = res.F
        return self

    def transform(self, features, permutation=None):
        if permutation is None:
            permutation = self.best_permutation
        reordered_features = features[:, permutation]
        return self.binarizer(reordered_features)