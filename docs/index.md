---
layout: main
---

# Vladimir Fanaskov

I am a researcher studying how deep learning can be applied in classical fields of numerical analysis, scientific computing, and the natural sciences. Curiosity often pulls me toward different corners of applied mathematics at once, usually in a haphazard way.

This page serves as both a personal introduction and an archive of public research notes. Expect regular updates of the content at least once a year.

* [Google Scholar](https://scholar.google.com/citations?user=iK5gdo8AAAAJ&hl=en)
* [arXiv](https://arxiv.org/search/math?searchtype=author&query=Fanaskov,+V)
* [GitHub](https://github.com/VLSF)

### What's New

{% for post in site.posts limit:5 %}
* **{{ post.date | date: "%b %d, %Y" }}**: [{{ post.title }}]({{ post.url }})
{% endfor %}

[See all updates](/archive.md)

Check out my [Projects](notes/projects.md).
