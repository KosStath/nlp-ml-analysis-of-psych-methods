import json
import numpy as np
import ijson  
from decimal import Decimal

# Updated term weights 
term_weights = {
    'aba-design': 1.0,
    'alternating-treatments-design': 1.0,
    'multiple-baseline-design': 1.0,
    'reversal-design': 1.0,
    'between-subjects-factorial-design': 1.0,
    'mixed-factorial-design': 1.0,
    'randomized-clinical-trial': 1.0,
    'double-blind-study': 1.0,
    'single-blind-study': 1.0,
    'anova': 1.0,
    'factorial-anova': 1.0,
    'repeated-measures-anova': 1.0,
    'factor-analysis': 1.0,
    'exploratory-factor-analysis': 1.0,
    'confirmatory-factor-analysis': 1.0,
    'meta-analysis': 1.0,
    'multiple-regression': 1.0,
    'simple-regression': 1.0,
    'chi-square-test-for-independence': 1.0,
    'chi-square-test-for-goodness-of-fit': 1.0,
    'cluster-sampling': 1.0,
    'stratified-random-sampling': 1.0,
    'proportionate-stratified-random-sampling': 1.0,
    'disproportionate-stratified-random-sampling': 1.0,
    'alpha': 0.8,
    'construct-validity': 0.8,
    'discriminant-validity': 0.8,
    'convergent-validity': 0.8,
    'criterion-validity': 0.8,
    'predictive-validity': 0.8,
    'test-retest-reliability': 0.8,
    'inter-rater-reliability': 0.8,
    'cronbach\'s-alpha': 0.8,
    'split-half-correlation': 0.8,
    'random-assignment': 0.8,
    'block-randomization': 0.8,
    'counterbalancing': 0.8,
    'complete-counterbalancing': 0.8,
    'matched-groups-design': 0.8,
    'pretest-posttest-design': 0.8,
    'interrupted-time-series-design': 0.8,
    'longitudinal-study': 0.8,
    'cross-sequential-study': 0.8,
    'correlation': 0.6,
    'pearson-correlation': 0.6,
    'pearson\'s-r': 0.6,
    't-test': 0.6,
    'independent-t-test': 0.6,
    'dependent-t-test': 0.6,
    'effect-size': 0.6,
    'cohen\'s-d': 0.6,
    'eta-squared': 0.6,
    'internal-validity': 0.6,
    'external-validity': 0.6,
    'statistical-validity': 0.6,
    'control-group': 0.6,
    'experimental-group': 0.6,
    'placebo-control': 0.6,
    'statistical-power': 0.6,
    'experimenter-bias': 0.6,
    'confounding-variable': 0.6,
    'sampling-bias': 0.6,
    'statistical-control': 0.6,
    'cross-sectional-study': 0.6,
    'power-analysis': 0.6,
    'histogram': 0.4,
    'scatterplot': 0.4,
    'bar-graph': 0.4,
    'line-graph': 0.4,
    'stem-and-leaf-graph': 0.4,
    'error-bars': 0.4,
    'standard-deviation': 0.4,
    'variance': 0.4,
    'confidence-interval': 0.4,
    'normal-distribution': 0.4,
    'z-distribution': 0.4,
    'chi-square-distribution': 0.4,
    'frequency-distribution': 0.4,
    'convenience-sampling': 0.4,
    'snowball-sampling': 0.4,
    'quota-sampling': 0.4,
    'demand-characteristics': 0.4,
    'hawthorne-effect': 0.4,
    'carryover-effect': 0.4,
    'order-effect': 0.4,
    'attrition': 0.4,
    'informed-consent': 0.4,
    'debriefing': 0.4,
    'pilot-test': 0.4,
    'manipulation-check': 0.4,
    'outlier': 0.2,
    'missing-data': 0.2,
    'statistical-independence': 0.2,
    'effect-coding': 0.2,
    'mixed-methods-research': 0.2,
    'qualitative-research': 0.2,
    'quantitative-research': 0.2,
    'observational-research': 0.2,
    'archival-research': 0.2,
    'systematic-error': 0.2,
    'random-error': 0.2,
    'parameter': 0.2,
    'probability-distribution': 0.2,
    'empirical-research': 0.2,
    'applied-research': 0.2 
}

def decimal_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError

def recalculate_abstract_embedding(abstract):
    term_embeddings = abstract.get('key_embeddings', [])
    
    if not term_embeddings:
        return [0.0] * 768

    weighted_embeddings = np.zeros((768,), dtype=np.float32)
    total_weight = 0.0
    
    for key_embedding in term_embeddings:
        for term, embeddings in key_embedding.items():
            if term in term_weights:
                weight = term_weights[term]
                if isinstance(embeddings, list) and all(isinstance(e, list) for e in embeddings):
                    for embedding in embeddings:
                        if len(embedding) == 768:
                            embedding = [float(value) for value in embedding]
                            weighted_embeddings += weight * np.array(embedding)
                            total_weight += weight
                        else:
                            print(f"Warning: Embedding for term '{term}' does not have 768 dimensions.")
                else:
                    print(f"Warning: Embedding for term '{term}' is not a valid list of lists.")

    if total_weight > 0:
        new_embedding = weighted_embeddings / total_weight
    else:
        new_embedding = np.zeros((768,), dtype=np.float32)
    
    return new_embedding.tolist()


# Placeholder paths
input_file = r'PATH_TO_YOUR_INPUT_FILE.json'
output_file = r'PATH_TO_YOUR_OUTPUT_FILE.json'

output_data = []

# Load JSON data using ijson
try:
    with open(input_file, 'r') as file:
        for idx, abstract in enumerate(ijson.items(file, 'item')):
            try:
                new_embedding = recalculate_abstract_embedding(abstract)
                abstract['embedding'] = new_embedding
                output_data.append(abstract)
                
                if idx % 1000 == 0:
                    print(f"Processed {idx} abstracts...")

            except Exception as e:
                print(f"Error processing abstract with id {abstract.get('id', 'unknown id')}: {e}")
        
        print(f"Total processed abstracts: {idx + 1}")

except Exception as e:
    print(f"Error reading input file: {e}")

try:
    with open(output_file, 'w') as file:
        json.dump(output_data, file, default=decimal_default, indent=4)  # Use the custom encoder
    print(f"Output saved to {output_file}")
except Exception as e:
    print(f"Error writing output file: {e}")
