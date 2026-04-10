---
layout: default
title: Archive
permalink: /archive/
---

# All Updates

<ul>
  {% for post in site.posts %}
    <li>
      {{ post.date | date: "%Y-%m-%d" }} —
      <a href="{{ post.url }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>

<br>
[Back to Home](/)
