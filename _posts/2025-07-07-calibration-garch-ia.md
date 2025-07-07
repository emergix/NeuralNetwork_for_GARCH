---
title: "Calibration GARCH et réseaux de neurones en finance"
layout: single
categories: [finance, machine-learning]
tags: [GARCH, neural-networks, calibration, quant]
---

Ce billet décrit comment une calibration GARCH intelligente peut être utilisée...

title: "NeuralNetwork_for_GARCH"
description: "GARCH + Deep Learning applications in quantitative finance"
remote_theme: "mmistakes/minimal-mistakes"
plugins:
  - jekyll-include-cache
  - jekyll-feed
  - jekyll-seo-tag

author:
  name: "Olivier Croissant"
  avatar: "/assets/images/avatar.png"  # (tu pourras l’ajouter plus tard)
  bio: "Quantitative finance, AI, and volatility modeling."
  location: "Paris, France"

markdown: kramdown
highlighter: rouge
theme: null

# Structure du site
defaults:
  - scope:
      path: ""
      type: "posts"
    values:
      layout: single
      author_profile: true
      read_time: true
      comments: false
      share: true
      related: true

# Navigation
include:
  - _pages

paginate: 5
paginate_path: /page:num/

