import { useState } from "react";

import "./App.css";

import ReactMarkdown from "react-markdown";

import remarkGfm from "remark-gfm";

//const API_BASE_URL = "http://127.0.0.1:8000";

const API_BASE_URL = "/api";

function App() {

  const [resume, setResume] = useState(null);

  const [jobDescription, setJobDescription] = useState("");

  const [loading, setLoading] = useState(false);

  const [downloading, setDownloading] = useState(false);

  const [error, setError] = useState("");

  const [extractedText, setExtractedText] = useState("");

  const [analysis, setAnalysis] = useState("");

  const [optimizedResume, setOptimizedResume] = useState(null);

  function handleResumeChange(event) {

    const file = event.target.files?.[0];

    setError("");

    setExtractedText("");

    setAnalysis("");

    setOptimizedResume(null);

    if (!file) {

      setResume(null);

      return;

    }

    if (

      file.type !== "application/pdf" &&

      !file.name.toLowerCase().endsWith(".pdf")

    ) {

      setError("Please select a PDF file.");

      event.target.value = "";

      setResume(null);

      return;

    }

    if (file.size > 5 * 1024 * 1024) {

      setError("Please select a PDF smaller than 5 MB.");

      event.target.value = "";

      setResume(null);

      return;

    }

    setResume(file);

  }

  async function readResponse(response) {

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {

      throw new Error(

        data.detail ||

          data.message ||

          `Request failed with status ${response.status}.`

      );

    }

    return data;

  }

  async function handleAnalyze() {

    if (!resume) {

      setError("Please upload your resume in PDF format.");

      return;

    }

    if (!jobDescription.trim()) {

      setError("Please enter the job description.");

      return;

    }

    setLoading(true);

    setError("");

    setExtractedText("");

    setAnalysis("");

    setOptimizedResume(null);

    try {

      // Step 1: Extract text from the uploaded PDF.

      const resumeForm = new FormData();

      resumeForm.append("file", resume);

      const extractResponse = await fetch(

        `${API_BASE_URL}/extract-resume`,

        {

          method: "POST",

          body: resumeForm,

        }

      );

      const extractData = await readResponse(extractResponse);

      if (!extractData.text?.trim()) {

        throw new Error("No readable text was extracted from your PDF.");

      }

      setExtractedText(extractData.text);

      // Prepare the resume and job description for both AI endpoints.

      const createResumeForm = () => {

        const form = new FormData();

        form.append("resume_text", extractData.text);

        form.append("job_description", jobDescription.trim());

        return form;

      };

      // Step 2: Generate the resume analysis.

      const analysisResponse = await fetch(

        `${API_BASE_URL}/analyze-resume`,

        {

          method: "POST",

          body: createResumeForm(),

        }

      );

      const analysisData = await readResponse(analysisResponse);

      if (!analysisData.analysis) {

        throw new Error("The AI did not return a resume analysis.");

      }

      setAnalysis(analysisData.analysis);

      // Step 3: Generate the rewritten, job-tailored resume.

      const optimizeResponse = await fetch(

        `${API_BASE_URL}/optimize-resume`,

        {

          method: "POST",

          body: createResumeForm(),

        }

      );

      const optimizeData = await readResponse(optimizeResponse);

      if (!optimizeData.optimized_resume?.trim()) {

        throw new Error(

          "The AI analysis succeeded, but no optimized resume was returned."

        );

      }

      setOptimizedResume(optimizeData);

    } catch (err) {

      if (err instanceof TypeError) {

        setError(

          "Cannot connect to the backend. Make sure FastAPI is running at http://127.0.0.1:8000."

        );

      } else {

        setError(err.message || "Something went wrong.");

      }

    } finally {

      setLoading(false);

    }

  }

  async function handleDownload() {

    if (!optimizedResume?.optimized_resume) {

      setError("Please generate an optimized resume first.");

      return;

    }

    setDownloading(true);

    setError("");

    try {

      const form = new FormData();

      form.append(

        "optimized_resume",

        optimizedResume.optimized_resume

      );

      const response = await fetch(

        `${API_BASE_URL}/download-optimized-resume`,

        {

          method: "POST",

          body: form,

        }

      );

      if (!response.ok) {

        const data = await response.json().catch(() => ({}));

        throw new Error(

          data.detail || "Unable to download the optimized resume."

        );

      }

      const blob = await response.blob();

      const url = window.URL.createObjectURL(blob);

      const link = document.createElement("a");

      link.href = url;

      link.download = "optimized_resume.pdf";

      document.body.appendChild(link);

      link.click();

      link.remove();

      window.URL.revokeObjectURL(url);

    } catch (err) {

      if (err instanceof TypeError) {

        setError(

          "Cannot connect to the backend to download your resume."

        );

      } else {

        setError(err.message || "Resume download failed.");

      }

    } finally {

      setDownloading(false);

    }

  }

  function handleReset() {

    setResume(null);

    setJobDescription("");

    setExtractedText("");

    setAnalysis("");

    setOptimizedResume(null);

    setError("");

    const fileInput = document.getElementById("resume");

    if (fileInput) {

      fileInput.value = "";

    }

  }

  return (

    <div className="app">

      <nav className="navbar">

        <a className="logo" href="/">

          ResumeAI

        </a>

        <span>AI-Powered Career Assistant</span>

      </nav>

      <main className="container">

        <div className="hero">

          <span className="badge">SMART CAREER ANALYSIS</span>

          <h1>

            Your next job starts

            <br />

            <span>with a better resume.</span>

          </h1>

          <p>

            Analyze your resume against a job description, discover

            missing skills, improve your resume, and prepare for

            interviews with AI assistance.

          </p>

        </div>

        <section className="card">

          <h2>Analyze & Optimize Your Resume</h2>

          <p className="description">

            Upload your resume and provide the target job description.

            ResumeAI will analyze your resume and create a tailored version.

          </p>

          <label htmlFor="resume">Upload Resume (PDF)</label>

          <div className="upload-box">

            <input

              id="resume"

              type="file"

              accept=".pdf,application/pdf"

              onChange={handleResumeChange}

              disabled={loading}

            />

            {resume && (

              <p className="file-name">

                Selected: {resume.name}

              </p>

            )}

          </div>

          <label htmlFor="job">Job Description</label>

          <textarea

            id="job"

            rows={7}

            placeholder="Paste the complete job description here..."

            value={jobDescription}

            onChange={(event) =>

              setJobDescription(event.target.value)

            }

            disabled={loading}

          />

          <button

            type="button"

            onClick={handleAnalyze}

            disabled={loading || downloading}

          >

            {loading

              ? "Analyzing & Optimizing..."

              : "Analyze & Optimize My Resume →"}

          </button>

          <p className="privacy-note">

            Your resume is sent to your backend and configured AI provider

            for analysis and optimization.

          </p>

        </section>

        {error && (

          <section className="card error-message" role="alert">

            <h2>Notice</h2>

            <p>{error}</p>

          </section>

        )}

        {loading && (

          <section className="card">

            <h2>Working on your resume...</h2>

            <p>

              Extracting your resume, analyzing the job requirements,

              and generating a tailored resume. This may take a little

              time.

            </p>

          </section>

        )}

        {analysis && (

          <section className="card">

            <h2>AI Resume Analysis</h2>

            <p className="description">

              Review your strengths, matching skills, skill gaps,

              improvement suggestions, and interview questions.

            </p>

            <div className="analysis-result">

              <div className="markdown-content">
  <ReactMarkdown remarkPlugins={[remarkGfm]}>{analysis}</ReactMarkdown>
</div>

            </div>

          </section>

        )}

        {optimizedResume && (

          <section className="card">

            <h2>Your Optimized Resume</h2>

            <p className="description">

              Your resume has been rewritten for the target job using

              the information in your original resume.

            </p>

            {typeof optimizedResume.ats_match_score === "number" && (

              <div className="score-panel">

                <h3>

                  Estimated ATS Match Score

                </h3>

                <p className="score-value">

                  {optimizedResume.ats_match_score}/100

                </p>

                <p>

                  {optimizedResume.score_explanation ||

                    "This is an estimated match based on the supplied job description."}

                </p>

              </div>

            )}

            <div className="optimized-resume">

              <h3>Rewritten Resume Preview</h3>

              <div className="markdown-content">
  <ReactMarkdown remarkPlugins={[remarkGfm]}>{optimizedResume.optimized_resume}</ReactMarkdown>
</div>

            </div>

            {Array.isArray(optimizedResume.changes_made) &&

              optimizedResume.changes_made.length > 0 && (

                <div className="optimization-details">

                  <h3>Changes Made</h3>

                  <ul>

                    {optimizedResume.changes_made.map(

                      (change, index) => (

                        <li key={index}>{change}</li>

                      )

                    )}

                  </ul>

                </div>

              )}

            {Array.isArray(optimizedResume.missing_skills) &&

              optimizedResume.missing_skills.length > 0 && (

                <div className="optimization-details">

                  <h3>Skills to Learn</h3>

                  <p className="description">

                    These are recommendations, not skills automatically

                    added to your existing qualifications.

                  </p>

                  <ul>

                    {optimizedResume.missing_skills.map(

                      (item, index) => (

                        <li key={index}>

                          <strong>

                            {typeof item === "string"

                              ? item

                              : item.skill}

                          </strong>

                          {typeof item === "object" && item.reason

                            ? ` — ${item.reason}`

                            : ""}

                        </li>

                      )

                    )}

                  </ul>

                </div>

              )}

            <button

              type="button"

              onClick={handleDownload}

              disabled={downloading || loading}

            >

              {downloading

                ? "Preparing Download..."

                : "Download Optimized Resume (PDF) ↓"}

            </button>

            <button

              type="button"

              onClick={handleReset}

              disabled={downloading || loading}

            >

              Analyze Another Resume

            </button>

            <p className="privacy-note">

              Review the generated resume carefully before applying.

              Verify that all details are accurate and that you can

              support every listed qualification.

            </p>

          </section>

        )}

        {extractedText && (

          <details className="card">

            <summary>View Extracted Resume Text</summary>

            <p className="description">

              Extracted {extractedText.length} characters from your PDF.

            </p>

            <textarea

              rows={12}

              value={extractedText}

              readOnly

              aria-label="Extracted resume text"

            />

          </details>

        )}

        <section className="features">

          <div className="feature">

            <span>01</span>

            <h3>Resume Analysis</h3>

            <p>Understand your resume's strengths.</p>

          </div>

          <div className="feature">

            <span>02</span>

            <h3>Resume Optimization</h3>

            <p>Tailor your resume to the target position.</p>

          </div>

          <div className="feature">

            <span>03</span>

            <h3>Interview Preparation</h3>

            <p>Prepare with relevant interview questions.</p>

          </div>

        </section>

        <footer>

          ResumeAI — Built with React, Python, FastAPI, and AI.

        </footer>

      </main>

    </div>

  );

}

export default App;
