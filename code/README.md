#!/usr/bin/env python3
"""
Nordico Paradigm - Voynich Manuscript Analysis Script
Version: v11.0
Author: Ben Yamoun Ali - Nordico Research Group
Date: 2025
Description: Reproducible quantitative analysis of the Voynich Manuscript
"""

import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - NORDEKO PARADIGM v11.0
# ============================================================================

class NordicoConfig:
    """Configuration class for Nordico Paradigm v11.0"""
    
    # Percentiles v7.4 (based on 64-folio training set)
    PERCENTILES = {
        'P_o': {'5th': 0.1408, '95th': 0.3900},
        'R_vc': {'5th': 0.6833, '95th': 1.1445}
    }
    
    # CI weights (validated by PCA)
    CI_WEIGHTS = {'P_o': 0.72, 'R_vc': 0.28}
    
    # Classification thresholds v11.0
    THRESHOLDS = {
        'extreme_alpha': 0.30,
        'alpha_beta': 0.56,
        'beta_extreme_beta': 0.80
    }
    
    # Functional systems definitions (>8% threshold)
    SYSTEMS = {
        'OTAL': ['otal', 'otar', 'otaim', 'otaiin', 'otol', 'otchy', 'otchor'],
        'CHOR': ['chol', 'chor', 'chey', 'cheey', 'chedy', 'chy', 'cthy'],
        'QOK': ['qok', 'qoke', 'qot', 'qo', 'qoky', 'qopchy', 'qokchy']
    }
    
    # EVA alphabet definitions
    VOWELS = set('aeioy')
    CONSONANTS = set('bcdfghjklmnpqrstuvwxz')
    GALLOWS = set('tkpf')  # EVA standard gallows characters
    
    @classmethod
    def get_P_o_range(cls):
        """Get normalized range for P_o"""
        return cls.PERCENTILES['P_o']['95th'] - cls.PERCENTILES['P_o']['5th']
    
    @classmethod
    def get_R_vc_range(cls):
        """Get normalized range for R_vc"""
        return cls.PERCENTILES['R_vc']['95th'] - cls.PERCENTILES['R_vc']['5th']

# ============================================================================
# CORE PROCESSING FUNCTIONS
# ============================================================================

def preprocess_text(text: str) -> List[str]:
    """
    Preprocess Voynich EVA transcription according to Nordico Protocol v11.0
    
    Steps:
    1. Remove editorial annotations: <...>
    2. Tokenize by '.' (EVA word separator)
    3. Filter alphabetic tokens (a-z only)
    
    Args:
        text: Raw EVA transcription
        
    Returns:
        List of valid tokens
    """
    # Step 1: Remove editorial annotations
    cleaned = re.sub(r'<[^>]+>', '', text)
    
    # Step 2: Tokenize by '.' (EVA word separator)
    raw_tokens = cleaned.split('.')
    
    # Step 3: Filter alphabetic tokens (a-z only)
    tokens = [t.lower() for t in raw_tokens if re.match(r'^[a-z]+$', t.lower())]
    
    return tokens

def calculate_P_o(tokens: List[str]) -> float:
    """
    Calculate P_o metric: frequency of 'o' + gallows characters
    
    P_o = (count('o') + count(gallows)) / total_chars
    
    Args:
        tokens: List of preprocessed tokens
        
    Returns:
        P_o value
    """
    if not tokens:
        return 0.0
    
    # Count characters
    text = ''.join(tokens)
    total_chars = len(text)
    
    if total_chars == 0:
        return 0.0
    
    # Count 'o' characters
    count_o = text.count('o')
    
    # Count gallows characters
    count_gallows = sum(text.count(c) for c in NordicoConfig.GALLOWS)
    
    # Calculate P_o
    P_o = (count_o + count_gallows) / total_chars
    
    return P_o

def calculate_R_vc(tokens: List[str]) -> float:
    """
    Calculate R_vc metric: vowel-consonant ratio
    
    R_vc = count(vowels) / count(consonants)
    
    Args:
        tokens: List of preprocessed tokens
        
    Returns:
        R_vc value (0 if no consonants)
    """
    if not tokens:
        return 0.0
    
    text = ''.join(tokens)
    
    # Count vowels and consonants
    count_vowels = sum(1 for c in text if c in NordicoConfig.VOWELS)
    count_consonants = sum(1 for c in text if c in NordicoConfig.CONSONANTS)
    
    # Avoid division by zero
    if count_consonants == 0:
        return 0.0
    
    return count_vowels / count_consonants

def normalize_value(value: float, metric: str) -> float:
    """
    Normalize metric using percentiles v7.4
    
    norm = clamp((value - 5th_percentile) / range, 0, 1)
    
    Args:
        value: Raw metric value
        metric: 'P_o' or 'R_vc'
        
    Returns:
        Normalized value [0, 1]
    """
    percentiles = NordicoConfig.PERCENTILES[metric]
    p5 = percentiles['5th']
    p95 = percentiles['95th']
    
    if p95 == p5:  # Avoid division by zero
        return 0.0
    
    normalized = (value - p5) / (p95 - p5)
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, normalized))

def calculate_CI(P_o: float, R_vc: float) -> float:
    """
    Calculate Continuum Index (CI) v7.4
    
    CI = 0.72 * P_o_norm + 0.28 * R_vc_norm
    
    Args:
        P_o: Raw P_o value
        R_vc: Raw R_vc value
        
    Returns:
        CI value [0, 1]
    """
    # Normalize metrics
    P_o_norm = normalize_value(P_o, 'P_o')
    R_vc_norm = normalize_value(R_vc, 'R_vc')
    
    # Calculate CI with validated weights
    CI = (NordicoConfig.CI_WEIGHTS['P_o'] * P_o_norm + 
          NordicoConfig.CI_WEIGHTS['R_vc'] * R_vc_norm)
    
    return CI

def calculate_functional_systems(tokens: List[str]) -> Dict[str, float]:
    """
    Calculate functional system frequencies
    
    Args:
        tokens: List of preprocessed tokens
        
    Returns:
        Dictionary with OTAL, CHOR, QOK frequencies (%)
    """
    if not tokens:
        return {'OTAL': 0.0, 'CHOR': 0.0, 'QOK': 0.0}
    
    total_tokens = len(tokens)
    
    # Initialize counters
    systems_count = {system: 0 for system in NordicoConfig.SYSTEMS.keys()}
    
    # Count occurrences for each system
    for token in tokens:
        for system, patterns in NordicoConfig.SYSTEMS.items():
            if any(token.startswith(pattern) or pattern in token 
                  for pattern in patterns):
                systems_count[system] += 1
                break  # Token can belong to multiple systems? We use first match
    
    # Convert to percentages
    systems_pct = {}
    for system, count in systems_count.items():
        systems_pct[system] = (count / total_tokens * 100) if total_tokens > 0 else 0.0
    
    return systems_pct

def calculate_affixes(tokens: List[str]) -> Dict[str, float]:
    """
    Calculate affix frequencies
    
    Args:
        tokens: List of preprocessed tokens
        
    Returns:
        Dictionary with affix frequencies (%)
    """
    if not tokens:
        return {
            'prefix_o': 0.0,
            'suffix_y': 0.0,
            'suffix_ain': 0.0,
            'suffix_dy': 0.0,
            'suffix_ol': 0.0
        }
    
    total_tokens = len(tokens)
    
    # Initialize counters
    affixes = {
        'prefix_o': 0,    # tokens starting with 'o'
        'suffix_y': 0,    # tokens ending with 'y'
        'suffix_ain': 0,  # tokens containing 'ain' or 'aiin'
        'suffix_dy': 0,   # tokens ending with 'dy'
        'suffix_ol': 0    # tokens ending with 'ol'
    }
    
    # Count affixes
    for token in tokens:
        if token.startswith('o'):
            affixes['prefix_o'] += 1
        if token.endswith('y'):
            affixes['suffix_y'] += 1
        if 'ain' in token or 'aiin' in token:
            affixes['suffix_ain'] += 1
        if token.endswith('dy'):
            affixes['suffix_dy'] += 1
        if token.endswith('ol'):
            affixes['suffix_ol'] += 1
    
    # Convert to percentages
    affixes_pct = {}
    for affix, count in affixes.items():
        affixes_pct[affix] = (count / total_tokens * 100) if total_tokens > 0 else 0.0
    
    return affixes_pct

def classify_regime(CI: float) -> str:
    """
    Classify folio into operational regime v11.0
    
    Args:
        CI: Continuum Index value
        
    Returns:
        Regime classification
    """
    thresholds = NordicoConfig.THRESHOLDS
    
    if CI < thresholds['extreme_alpha']:
        return "α extreme"
    elif CI < thresholds['alpha_beta']:
        return "α standard"
    elif CI < thresholds['beta_extreme_beta']:
        return "β"
    else:
        return "β extreme"

def analyze_folio(folio_text: str, folio_id: str = "", 
                  quire: str = "", illustration: str = "") -> Dict:
    """
    Complete analysis of a single Voynich folio
    
    Args:
        folio_text: EVA transcription
        folio_id: Folio identifier (e.g., 'f1r')
        quire: Quire identifier
        illustration: Type of illustration
        
    Returns:
        Complete analysis results
    """
    # Step 1: Preprocessing
    tokens = preprocess_text(folio_text)
    
    # Step 2: Calculate core metrics
    P_o = calculate_P_o(tokens)
    R_vc = calculate_R_vc(tokens)
    
    # Step 3: Normalization
    P_o_norm = normalize_value(P_o, 'P_o')
    R_vc_norm = normalize_value(R_vc, 'R_vc')
    
    # Step 4: Calculate CI
    CI = calculate_CI(P_o, R_vc)
    
    # Step 5: Functional systems
    systems = calculate_functional_systems(tokens)
    
    # Step 6: Affix analysis
    affixes = calculate_affixes(tokens)
    
    # Step 7: Classification
    regime = classify_regime(CI)
    
    # Determine Currier classification based on CHOR
    currier_class = "A" if systems.get('CHOR', 0) > 8.0 else "B"
    
    # Compile results
    results = {
        'folio': folio_id,
        'quire': quire,
        'illustration': illustration,
        'characters': sum(len(t) for t in tokens),
        'tokens': len(tokens),
        'P_o': float(P_o),
        'R_vc': float(R_vc),
        'P_o_norm': float(P_o_norm),
        'R_vc_norm': float(R_vc_norm),
        'CI': float(CI),
        'systems': systems,
        'affixes': affixes,
        'regime': regime,
        'currier_class': currier_class,
        'type_provisoire': determine_provisional_type(systems, CI, illustration)
    }
    
    return results

def determine_provisional_type(systems: Dict, CI: float, 
                               illustration: str) -> str:
    """
    Determine provisional folio type based on analysis
    
    Args:
        systems: Functional system frequencies
        CI: Continuum Index
        illustration: Illustration type
        
    Returns:
        Provisional type label
    """
    # Map illustration to type
    illustration_map = {
        'botanical': 'botanical',
        'herbal': 'botanical',
        'zodiacal': 'zodiacal',
        'astronomical': 'astral',
        'cosmological': 'cosmological',
        'biological': 'biological',
        'pharmaceutical': 'pharmaceutical',
        'stars': 'astral',
        'diagram': 'diagrammatic'
    }
    
    base_type = illustration_map.get(illustration.lower(), 'textual')
    
    # Add regime information
    regime = classify_regime(CI)
    
    # Add system information
    if systems.get('OTAL', 0) > 20:
        qualifier = 'nominal'
    elif systems.get('CHOR', 0) > 20:
        qualifier = 'descriptive'
    elif systems.get('QOK', 0) > 20:
        qualifier = 'procedural'
    else:
        qualifier = 'mixed'
    
    return f"{base_type} {regime} {qualifier}"

# ============================================================================
# BATCH PROCESSING AND STATISTICAL ANALYSIS
# ============================================================================

class NordicoAnalyzer:
    """Main analyzer class for batch processing"""
    
    def __init__(self, config: NordicoConfig = None):
        self.config = config or NordicoConfig()
        self.results = []
        
    def analyze_corpus(self, corpus_data: List[Dict]) -> pd.DataFrame:
        """
        Analyze entire corpus of folios
        
        Args:
            corpus_data: List of dictionaries with folio data
                Each dict should have: 'folio', 'text', 'quire', 'illustration'
                
        Returns:
            DataFrame with all analysis results
        """
        print(f"Analyzing {len(corpus_data)} folios...")
        
        results = []
        for folio_data in corpus_data:
            result = analyze_folio(
                folio_text=folio_data.get('text', ''),
                folio_id=folio_data.get('folio', ''),
                quire=folio_data.get('quire', ''),
                illustration=folio_data.get('illustration', '')
            )
            results.append(result)
        
        self.results = results
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Add additional derived metrics
        df['is_currier_A'] = df['currier_class'] == 'A'
        df['total_systems'] = df['systems'].apply(
            lambda x: sum(1 for v in x.values() if v > 8)
        )
        
        return df
    
    def calculate_global_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate global statistics across corpus"""
        stats = {
            'total_folios': len(df),
            'total_tokens': df['tokens'].sum(),
            'total_characters': df['characters'].sum(),
            
            'CI_statistics': {
                'mean': df['CI'].mean(),
                'std': df['CI'].std(),
                'min': df['CI'].min(),
                'max': df['CI'].max(),
                'median': df['CI'].median()
            },
            
            'regime_distribution': df['regime'].value_counts().to_dict(),
            'currier_distribution': df['currier_class'].value_counts().to_dict(),
            
            'quire_statistics': df.groupby('quire')['CI'].agg(['mean', 'std', 'count']).to_dict('index')
        }
        
        return stats
    
    def identify_extreme_alpha(self, df: pd.DataFrame, 
                               threshold: float = 0.30) -> pd.DataFrame:
        """Identify extreme α folios (CI < threshold)"""
        extreme_alpha = df[df['CI'] < threshold].copy()
        extreme_alpha = extreme_alpha.sort_values('CI')
        return extreme_alpha
    
    def analyze_bifolio_coherence(self, df: pd.DataFrame, 
                                  bifolio_data: Dict) -> Dict:
        """
        Analyze intra-bifolio stylistic coherence
        
        Args:
            df: Analysis results DataFrame
            bifolio_data: Dict mapping bifolio_id to list of folio_ids
            
        Returns:
            Coherence analysis results
        """
        coherence_results = []
        
        for bifolio_id, folio_ids in bifolio_data.items():
            bifolio_folios = df[df['folio'].isin(folio_ids)]
            
            if len(bifolio_folios) < 2:
                continue
            
            # Check if all folios in bifolio share same regime
            regimes = bifolio_folios['regime'].unique()
            is_coherent = len(regimes) == 1
            
            coherence_results.append({
                'bifolio_id': bifolio_id,
                'folios': list(bifolio_folios['folio']),
                'regimes': list(regimes),
                'is_coherent': is_coherent,
                'mean_CI': bifolio_folios['CI'].mean(),
                'CI_std': bifolio_folios['CI'].std()
            })
        
        # Calculate overall coherence
        total_bifolios = len(coherence_results)
        coherent_bifolios = sum(1 for r in coherence_results if r['is_coherent'])
        coherence_rate = coherent_bifolios / total_bifolios if total_bifolios > 0 else 0
        
        return {
            'coherence_rate': coherence_rate,
            'coherent_bifolios': coherent_bifolios,
            'total_bifolios': total_bifolios,
            'details': coherence_results
        }
    
    def export_results(self, df: pd.DataFrame, output_dir: str = 'output'):
        """Export all results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. Full results CSV
        df.to_csv(output_path / 'nordico_full_results.csv', index=False)
        
        # 2. JSON export for each folio
        json_results = []
        for _, row in df.iterrows():
            folio_result = row.to_dict()
            # Convert numpy types to Python native
            for key, value in folio_result.items():
                if isinstance(value, (np.integer, np.floating)):
                    folio_result[key] = value.item()
            
            json_results.append(folio_result)
        
        with open(output_path / 'nordico_results.json', 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        # 3. Summary statistics
        stats = self.calculate_global_statistics(df)
        with open(output_path / 'nordico_statistics.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # 4. Extreme alpha folios
        extreme_alpha = self.identify_extreme_alpha(df)
        extreme_alpha.to_csv(output_path / 'extreme_alpha_folios.csv', index=False)
        
        print(f"Results exported to {output_path.absolute()}")

# ============================================================================
# VALIDATION AND VISUALIZATION FUNCTIONS
# ============================================================================

def validate_CI_weights(df: pd.DataFrame) -> Dict:
    """
    Validate CI weights using Random Forest feature importance
    
    Returns empirical ratio P_o:R_vc
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    
    # Prepare features
    X = df'P_o_norm', 'R_vc_norm'
    y = df['CI']
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
    
    # Fit on full data for feature importance
    rf.fit(X, y)
    importances = rf.feature_importances_
    
    # Calculate empirical ratio
    empirical_ratio = importances[0] / importances[1] if importances[1] > 0 else 0
    
    return {
        'empirical_ratio_Po_to_Rvc': float(empirical_ratio),
        'theoretical_ratio': 0.72 / 0.28,
        'cv_r2_mean': float(cv_scores.mean()),
        'cv_r2_std': float(cv_scores.std()),
        'feature_importances': {
            'P_o_norm': float(importances[0]),
            'R_vc_norm': float(importances[1])
        }
    }

def perform_statistical_tests(df: pd.DataFrame) -> Dict:
    """Perform statistical tests on the data"""
    from scipy import stats
    from scipy.stats import f_oneway
    
    results = {}
    
    # 1. Test for unimodality (simplified)
    ci_values = df['CI'].values
    dip_stat, p_value = stats.diptest(ci_values)
    results['unimodality_test'] = {
        'dip_statistic': float(dip_stat),
        'p_value': float(p_value),
        'is_unimodal': p_value > 0.05
    }
    
    # 2. ANOVA across quires
    quire_groups = [group['CI'].values for name, group in df.groupby('quire')]
    if len(quire_groups) > 1:
        f_stat, p_val = f_oneway(*quire_groups)
        results['anova_across_quires'] = {
            'f_statistic': float(f_stat),
            'p_value': float(p_val),
            'significant': p_val < 0.05
        }
    
    # 3. Correlation between CI and systems
    for system in ['OTAL', 'CHOR', 'QOK']:
        system_values = df['systems'].apply(lambda x: x.get(system, 0))
        corr, p_val = stats.pearsonr(df['CI'], system_values)
        results[f'correlation_CI_{system}'] = {
            'correlation': float(corr),
            'p_value': float(p_val)
        }
    
    return results

# ============================================================================
# EXAMPLE USAGE AND DEMONSTRATION
# ============================================================================

def example_usage():
    """Example of how to use the Nordico analyzer"""
    
    # Sample corpus data (replace with your actual data)
    sample_corpus = [
        {
            'folio': 'f1r',
            'text': 'okainy.chey.dar.otal.cheey.qok.dy',  # Example EVA text
            'quire': 'A',
            'illustration': 'botanical'
        },
        {
            'folio': 'f1v',
            'text': 'chol.chor.otalim.qoke.dar.ain.ol',
            'quire': 'A',
            'illustration': 'botanical'
        },
        # Add more folios here...
    ]
    
    # Initialize analyzer
    analyzer = NordicoAnalyzer()
    
    # Analyze corpus
    results_df = analyzer.analyze_corpus(sample_corpus)
    
    # Calculate statistics
    stats = analyzer.calculate_global_statistics(results_df)
    
    # Export results
    analyzer.export_results(results_df)
    
    # Perform validation
    validation = validate_CI_weights(results_df)
    
    # Statistical tests
    stats_tests = perform_statistical_tests(results_df)
    
    return {
        'results': results_df,
        'statistics': stats,
        'validation': validation,
        'statistical_tests': stats_tests
    }

def load_corpus_from_json(json_path: str) -> List[Dict]:
    """Load corpus data from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def main():
    """Main function for running the Nordico analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Nordico Paradigm: Voynich Manuscript Analysis'
    )
    parser.add_argument('--input', type=str, 
                       help='Path to JSON file with corpus data')
    parser.add_argument('--output', type=str, default='nordico_output',
                       help='Output directory')
    parser.add_argument('--example', action='store_true',
                       help='Run example analysis')
    
    args = parser.parse_args()
    
    if args.example:
        print("Running example analysis...")
        results = example_usage()
        print("Example completed. Check 'output' directory for results.")
        return
    
    if args.input:
        print(f"Loading corpus from {args.input}...")
        corpus_data = load_corpus_from_json(args.input)
        
        analyzer = NordicoAnalyzer()
        results_df = analyzer.analyze_corpus(corpus_data)
        analyzer.export_results(results_df, args.output)
        
        print(f"Analysis complete. Results saved to {args.output}/")
        print(f"Analyzed {len(results_df)} folios.")
        print(f"Mean CI: {results_df['CI'].mean():.3f}")
        print(f"Regime distribution: {results_df['regime'].value_counts().to_dict()}")
    else:
        print("Please provide an input file or use --example flag")
        parser.print_help()

# ============================================================================
# DATA STRUCTURE FOR CORPUS INPUT
# ============================================================================

"""
Expected JSON structure for corpus input:

[
  {
    "folio": "f1r",
    "text": "okainy.chey.dar.otal.cheey.qok.dy",
    "quire": "A",
    "illustration": "botanical",
    "bifolio": "bA1"
  },
  {
    "folio": "f1v",
    "text": "chol.chor.otalim.qoke.dar.ain.ol",
    "quire": "A",
    "illustration": "botanical",
    "bifolio": "bA1"
  },
  ...
]
"""

# ============================================================================
# RUN SCRIPT
# ============================================================================

if __name__ == "__main__":
    main()
