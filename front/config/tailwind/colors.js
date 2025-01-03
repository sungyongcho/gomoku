import Color from "color";

export const generateShades = (name, color) => {
  return {
    [name]: Color(color).hex(),
    [`${name}-100`]: Color(color).lighten(0.5).hex(),
    [`${name}-200`]: Color(color).lighten(0.4).hex(),
    [`${name}-300`]: Color(color).lighten(0.3).hex(),
    [`${name}-400`]: Color(color).lighten(0.2).hex(),
    [`${name}-500`]: Color(color).hex(),
    [`${name}-600`]: Color(color).darken(0.2).hex(),
    [`${name}-700`]: Color(color).darken(0.3).hex(),
    [`${name}-800`]: Color(color).darken(0.4).hex(),
    [`${name}-900`]: Color(color).darken(0.5).hex(),
  };
};
