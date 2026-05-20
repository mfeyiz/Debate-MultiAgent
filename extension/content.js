(() => {
  const selectionText = String(window.getSelection?.() || "").trim();
  const article = document.querySelector("article");
  const main = document.querySelector("main");
  const metaDescription = document.querySelector('meta[name="description"]')?.content || "";
  const title = document.querySelector("h1")?.innerText || document.title || "";
  const source = article || main || document.body;

  const paragraphs = [...source.querySelectorAll("p, h1, h2, blockquote")]
    .map((node) => node.innerText.trim())
    .filter((text) => text.split(/\s+/).length >= 4)
    .filter((text, index, all) => all.indexOf(text) === index);

  const articleText = [title, metaDescription, ...paragraphs].join("\n\n").trim();

  return {
    url: location.href,
    title,
    selectionText,
    articleText,
  };
})();
