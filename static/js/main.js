document.addEventListener('DOMContentLoaded', function() {
    const analysisMode = document.getElementById('analysis_mode');
    const singleJdMultipleResumes = document.getElementById('single_jd_multiple_resumes');
    const multipleJdsSingleResume = document.getElementById('multiple_jds_single_resume');
    const batchProcessing = document.getElementById('batch_processing');

    analysisMode.addEventListener('change', function() {
        singleJdMultipleResumes.style.display = 'none';
        multipleJdsSingleResume.style.display = 'none';
        batchProcessing.style.display = 'none';

        switch(this.value) {
            case 'single_jd_multiple_resumes':
                singleJdMultipleResumes.style.display = 'block';
                break;
            case 'multiple_jds_single_resume':
                multipleJdsSingleResume.style.display = 'block';
                break;
            case 'batch_processing':
                batchProcessing.style.display = 'block';
                break;
        }
    });

    const form = document.getElementById('analyzeForm');
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        const formData = new FormData(this);

        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            displayResults(data.results);
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('results').innerHTML = '<p>An error occurred while processing your request.</p>';
        });
    });

    function displayResults(results) {
        const resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = '';

        results.forEach(result => {
            const resultElement = document.createElement('div');
            resultElement.classList.add('result-item', 'mb-8');

            resultElement.innerHTML = `
                <h3 class="text-xl font-semibold mb-4">Job Description: ${result.job_description}</h3>
                <h4 class="text-lg font-medium mb-4">Resume: ${result.resume}</h4>
                <div class="analysis-content">
                    ${result.analysis}
                </div>
                <hr class="my-8">
            `;
            resultsDiv.appendChild(resultElement);
        });
    }
});
