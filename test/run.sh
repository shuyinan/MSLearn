#!/bin/bash
# Script to run all tests for MSLearn

# Define test directories (one per line for easy maintenance)
TEST_DIRS=(
  "sample1_pip_nn"
  "sample2_pip_nn_pm"
  "sample3_pr_nn"
  "sample4_graph_mpnn"
  "sample5_graph_mpnn"
  "sample6_graph_mpnn_pm"
  "sample7_general_nn"
  "sample8_graph_mpnn_pm"
)

# Function to run a test in a given directory
run_test() {
  local test_dir=$1
  
  echo "🔹 Running test in: ${test_dir}"

  if [ ! -d "${test_dir}" ]; then
    echo "❌ ERROR: Directory ${test_dir} does not exist. Skipping..."
    return
  fi

  cd "${test_dir}" || { echo "❌ ERROR: Failed to enter ${test_dir}"; return; }

  # Clean old output and copy new input
  rm -rf output
  cp -rf input output

  cd output || { echo "❌ ERROR: Failed to enter output directory in ${test_dir}"; return; }

  # Run the training script and log output
  echo "🚀 Running mslearn-train in ${test_dir}..."
  mslearn-train config_test.yaml > log 2>&1

  # Check if the calculation was successful
  if grep -q "===All training outputs saved successfully!===" log; then
    echo "✅ SUCCESS: Training completed successfully in ${test_dir}"
  else
    echo "⚠️ WARNING: Training may have failed in ${test_dir}. Check log for details."
  fi

  cd ../../
}

# Run tests sequentially
for test_dir in "${TEST_DIRS[@]}"; do
  run_test "${test_dir}"
done

echo "🎉 All tests completed!"

