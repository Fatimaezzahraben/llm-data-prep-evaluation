# scaffold_project.ps1
# Run this from the root of your project: D:\MASTER_PFA\llm-data-prep-evaluation
# Creates the full folder/file skeleton. Safe to re-run (won't overwrite existing files).

$folders = @(
    "app\api",
    "app\services\evaluation",
    "app\models",
    "app\utils",
    "benchmark\datasets\hotel_bookings",
    "benchmark\datasets\adult_income",
    "benchmark\datasets\titanic",
    "benchmark\datasets\customer",
    "benchmark\datasets\wine_quality",
    "benchmark\tasks",
    "prompts",
    "workflows\baseline",
    "workflows\generated",
    "workflows\executed",
    "workflows\failed",
    "results\cleaned_datasets",
    "results\generated_workflows",
    "results\execution_logs",
    "results\metrics",
    "results\figures",
    "results\reports",
    "results\traceability",
    "results\reproducibility",
    "results\comparisons",
    "notebooks",
    "report",
    "tests",
    "docs\images",
    "scripts"
)

$files = @(
    "app\main.py",
    "app\api\health.py",
    "app\api\profile.py",
    "app\api\prompt.py",
    "app\api\workflow.py",
    "app\api\execution.py",
    "app\api\evaluation.py",
    "app\api\benchmark.py",
    "app\api\reports.py",
    "app\services\error_injection.py",
    "app\services\profiler.py",
    "app\services\prompt_builder.py",
    "app\services\llm.py",
    "app\services\workflow_generator.py",
    "app\services\safe_executor.py",
    "app\services\benchmark_manager.py",
    "app\services\evaluation\quality.py",
    "app\services\evaluation\cost.py",
    "app\services\evaluation\latency.py",
    "app\services\evaluation\robustness.py",
    "app\services\evaluation\traceability.py",
    "app\services\evaluation\reproducibility.py",
    "app\services\evaluation\metrics.py",
    "app\models\request_models.py",
    "app\models\response_models.py",
    "app\models\schemas.py",
    "app\utils\config.py",
    "app\utils\constants.py",
    "app\utils\logger.py",
    "app\utils\file_manager.py",
    "app\utils\helpers.py",
    "benchmark\benchmark_config.json",
    "prompts\system_prompt.txt",
    "prompts\prompt_simple.txt",
    "prompts\prompt_schema.txt",
    "prompts\prompt_profile.txt",
    "prompts\prompt_constrained.txt",
    "prompts\prompt_fewshot.txt",
    "prompts\prompt_validation_loop.txt",
    "notebooks\01_dataset_selection.ipynb",
    "notebooks\02_error_injection.ipynb",
    "notebooks\03_data_profiling.ipynb",
    "notebooks\04_prompt_engineering.ipynb",
    "notebooks\05_workflow_generation.ipynb",
    "notebooks\06_workflow_execution.ipynb",
    "notebooks\07_metrics_evaluation.ipynb",
    "notebooks\08_error_analysis.ipynb",
    "notebooks\09_visualizations.ipynb",
    "report\bibliographie.md",
    "report\proposition_these.md",
    "tests\test_profiler.py",
    "tests\test_prompt_builder.py",
    "tests\test_llm.py",
    "tests\test_executor.py",
    "tests\test_metrics.py",
    "tests\test_api.py",
    "docs\architecture.md",
    "docs\methodology.md",
    "docs\benchmark.md",
    "docs\api.md",
    "docs\installation.md",
    "scripts\create_noisy_dataset.py",
    "scripts\generate_all_workflows.py",
    "scripts\execute_all.py",
    "scripts\evaluate_all.py",
    "scripts\export_results.py",
    "config.yml",
    ".env",
    "README.md",
    "requirements.txt"
)

Write-Host "Creating folders..." -ForegroundColor Cyan
foreach ($f in $folders) {
    if (-not (Test-Path $f)) {
        New-Item -ItemType Directory -Path $f -Force | Out-Null
        Write-Host "  created: $f"
    }
}

Write-Host "Creating files..." -ForegroundColor Cyan
foreach ($f in $files) {
    if (-not (Test-Path $f)) {
        New-Item -ItemType File -Path $f -Force | Out-Null
        Write-Host "  created: $f"
    }
}

Write-Host "Done. Project structure scaffolded." -ForegroundColor Green
