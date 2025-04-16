+++
title = "My First Post"
date = "2025-02-18T18:33:43+03:00"
#dateFormat = "2006-01-02" # This value can be configured for per-post date formatting
author = "n"
authorTwitter = "nomo" #do not include @
cover = ""
tags = ["", ""]
keywords = ["", ""]
description = ""
showFullContent = false
readingTime = false
hideComments = false
[params]
    math = true
+++


## Introduction

This is **bold** text, and this is *emphasized* text.

Visit the [MaMen](https://dogukannn.github.io) website!

{{< code language="cpp" >}}

float sin_theta_max = radius / glm::length(intersection - center);
float cos_theta_max = std::sqrt(std::max(0.0f, 1.0f - sin_theta_max * sin_theta_max));

//create a basis and sample a position
vec3 wtheta = glm::normalize(center - intersection);
vec3 np = create_non_colinear_vector(wtheta);
vec3 u = glm::normalize(cross(wtheta, np));
vec3 v = glm::normalize(cross(wtheta, u));

float ch1 = frandom();
float ch2 = frandom();

float theta = std::acos(1.0f - ch1 + ch1 * cos_theta_max);
float phi = 2.0f * pi * ch2;

vec3 l = glm::normalize(u * std::cos(phi) * std::sin(theta) + v * std::sin(phi) * std::sin(theta) + wtheta * std::cos(theta));
{{< /code >}}


$$\int_0^1 \frac{1}{x} \, dx$$
$$\int_0^1 \frac{1}{x} \, dx$$
more different latex examples with integrals and summations:
$$\sum{i=1}^n i = \frac{n(n+1)}{2}$$
integral zero to inf
$$\int_0^\infty e^{-x} \, dx = 1$$


This is an inline \(a^*=x-b^*\) equation.

These are block equations:

\[a^*=x-b^*\]

\[ a^*=x-b^* \]

\[
a^*=x-b^*
\]

These are also block equations:

$$a^*=x-b^*$$

$$ a^*=x-b^* $$

$$
a^*=x-b^*
$$
