document.addEventListener("DOMContentLoaded", async () => {
  const pageTitle = document.getElementById("page-title");
  const content = document.getElementById("content");

  pageTitle.textContent = document.title;

  const sections = [
    "01_overview",
    "02_background",
    "03_litreview",
    "04_problem",
    "05_datasets",
    "06_methods",
    "07_results",
    "08_timeline",
    "09_references"
  ];

  content.innerHTML = "";

  for (const section of sections) {
    try {
      const response = await fetch(`sections/${section}.html`);

      if (!response.ok) {
        throw new Error(`Failed to load ${section}.html`);
      }

      const html = await response.text();
      const wrapper = document.createElement("div");
      wrapper.className = "section-wrapper";
      wrapper.innerHTML = html;
      content.appendChild(wrapper);
    } catch (error) {
      const errorBlock = document.createElement("section");
      errorBlock.className = "card";
      errorBlock.innerHTML = `
        <h2>Section load error</h2>
        <p>Could not load <code>${section}.html</code>.</p>
      `;
      content.appendChild(errorBlock);
      console.error(error);
    }
  }
});